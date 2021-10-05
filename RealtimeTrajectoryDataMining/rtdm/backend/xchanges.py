import itertools
import logging
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Union

import redis  # pip install redis-py

# Variables used for internal namespacing in class 'InMemoryMessageExchange'.
# They are necessary for distinguishing between kv store operations and queue
# operations
KVSTORE_IDENTIFIER: str = "kvstore"
QUEUE_IDENTIFIER: str = "queue"


class RedisMessageExchange:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 1,
        timeout: int = 5,
        max_try: int = 5,
        wait_time: float = 0.001,
    ) -> None:
        """
        Initialize and set given class variables on class instantiation.

        Args:
            host (str, optional): The service name or address of where Redis \
                is running. Defaults to "localhost".
            port (str, optional): The port number of the address where Redis \
                is running. Defaults to "6379".
            db (int, optional): The index of the Redis database that should \
                be used. Defaults to 1.
            timeout (int, optional): An upper bound on the amount of time we \
                are prepared to wait on the retrieval of a value before we \
                return with an error. Defaults to 5.
            max_try (int, optional): An upper bound on the number of \
                connection retries. Defaults to 5.
            wait_time (float, optional): The time between trying to SET or GET \
                values in the Redis KV store. Defaults to 0.001.
        """
        # Redis KV store connection info
        self.host = host
        self.port = port
        self.db = db
        # Redis connection status
        self.server_down: bool = True
        # The upper bound on number of re-tries
        self.max_try = max_try
        # The waiting time between re-tries, when processing data
        self.wait_time = wait_time
        # The timeout value used when trying to retrieve a value from Redis
        self.timeout = timeout
        # The backoff counter used to increase the waiting time
        # between re-triees, when trying to re-connect to Redis
        self.backoff: int = 0
        # The client connection
        self.client: Union[None, redis.Redis] = None
        # Check that the right args were given
        self.check_base_args()
        # Finally, connect to the Redis server
        self.connect()

    def check_base_args(self) -> None:
        """
        Check and validate all class arguments on calss instantiation.

        Raises:
            TypeError: Given input is NOT of type 'float'.
            TypeError: Given input is NOT of type 'int'.
            TypeError: Given input is NOT of type 'float'.
            TypeError: Given input is NOT of type 'int'.
        """
        if not isinstance(self.timeout, float):
            error = f"ARG 'timeout' is of type {type(self.timeout)} \
                but should be of type 'float'!"
            try:
                self.timeout = float(self.timeout)
            except Exception:
                raise TypeError(error)
        if not isinstance(self.max_try, int):
            error = f"ARG 'max_try' is of type {type(self.max_try)} \
                but should be of type 'int'!"
            try:
                self.max_try = int(self.max_try)
            except Exception:
                raise TypeError(error)
        if not isinstance(self.wait_time, float):
            error = f"ARG 'wait_time' is of type {type(self.wait_time)} \
                but should be of type 'float'!"
            try:
                self.wait_time = float(self.wait_time)
            except Exception:
                raise TypeError(error)

    def connect(self) -> bool:
        """
        Connect to the Redis and handle connection errors.

        Returns:
            bool: A boolean value that signals whether it was possible to \
                connect to Redis.
        """
        connection_string = (
            str(self.host) + ":" + str(self.port) + "/" + str(self.db)
        )
        # Assume we are not able to connect.
        # This flag is set to 'True' if we are able to connect successfully.
        return_value = False
        while self.server_down:
            try:
                self._connect()
                self.server_down = False
                self.backoff = 0  # Reset the backoff value
                logging.debug(
                    f"Client connection to {connection_string} is up!"
                )
                return_value = True
                break
            except redis.exceptions.ConnectionError:
                self.server_down = True
                # After each connection retry increase the backoff value
                self.backoff += 1
                # Increase the waiting time before we try to reconnect to the
                # Redis KV store.
                sleep_time = 3 * self.backoff
                time.sleep(sleep_time)
                logging.debug(
                    f"Cannot connect to {connection_string} trying again "
                    + f"in {sleep_time} seconds..."
                )
            if self.backoff == self.max_try:
                logging.debug(
                    f"No connection could be established to \
                    {connection_string} after {self.backoff} re-tries!"
                )
                break
        return return_value

    def reset_connection(self) -> bool:
        """
        Try reconecting to Redis.

        Raises:
            redis.exceptions.ConnectionError: The reconnection attempt did not \
                work so raise a Redis 'ConnectionError'.

        Returns:
            bool: In case the reconnection attempt succeeded then return 'True'.
        """
        # After 3 seconds try to re-connect...
        time.sleep(3)
        self.server_down = True
        is_connected = self.connect()
        if not is_connected:
            connection_string = (
                str(self.host) + ":" + str(self.port) + "/" + str(self.db)
            )
            logging.debug(
                f"Server is down. No connection could be established to \
                {connection_string}!"
            )
            raise redis.exceptions.ConnectionError
        else:
            return True

    def kv_set(
        self,
        data: Any,
        key: str,
        ttl: Union[None, int] = None,
        timeout: Union[None, float] = None,
    ) -> bool:
        """
        SET data in Redis.

        Args:
            data (Any): The data that should be SET in Redis and that also \
                contains a unique identifier 'message_uuid' that can be used \
                to retrieve a possible result.
            key (str): The key that the data should be associated with.
            ttl (Union[None, int], optional): The time before the data is \
                automatically expired. Defaults to None.
            timeout (Union[None, float], optional): An upper bound on the \
                amount of time we are prepared to wait on the retrieval of \
                a value before we return with an error. Defaults to None.

        Raises:
            TypeError: If the 'ttl' is not None or an integer value.
        """
        if timeout is None:
            timeout = self.timeout
        value_added = False
        if ttl is not None and ttl is not isinstance(ttl, int):
            error = (
                f"ARG 'ttl' is of type {type(ttl)} but should be of type 'int'!"
            )
            try:
                ttl = int(ttl)
            except Exception:
                raise TypeError(error)
        start_time = datetime.utcnow() + timedelta(seconds=self.timeout)
        while True:
            try:
                if self.client is not None:
                    return_code = self.client.set(key, data, ex=ttl,)
                    if return_code is True:
                        value_added = True
                        break
                else:
                    logging.warning("Attribute 'client' is None.")
            except redis.exceptions.ConnectionError:
                # Try to fix the connection
                self.reset_connection()
            if datetime.utcnow() - start_time > timedelta(seconds=self.timeout):
                logging.debug(
                    f"Waited {timeout} seconds. No message was set in Redis!"
                )
                break
            time.sleep(self.wait_time)
        return value_added

    def kv_get(
        self, key: str, timeout: Union[None, float] = None,
    ) -> Union[None, Any]:
        """
        GET data in Redis.

        Args:
            key (str): The key that the data is associated with.
            timeout (Union[None, float], optional): An upper bound on the \
                amount of time we are prepared to wait on the retrieval of \
                a value before we return with an error. Defaults to None.

        Returns:
            Any: The data retrieved from Redis.
        """
        if timeout is None:
            timeout = self.timeout
        start_time = datetime.utcnow() + timedelta(seconds=timeout)
        response = None
        # Get response data
        while True:
            try:
                if self.client is not None:
                    response = self.client.get(key,)  # noqa
                    if response is not None:
                        break
                else:
                    logging.warning("Attribute 'client' is None.")
            except redis.exceptions.ConnectionError:
                # Try to fix the connection
                self.reset_connection()
            if datetime.utcnow() - start_time > timedelta(seconds=timeout):
                logging.debug(
                    f"Waited {timeout} seconds. No message was returned!"
                )
                break
            time.sleep(self.wait_time)
        return response

    def _connect(self) -> None:
        """Internal method to check if the connection to Redis is working."""
        self.client = redis.Redis(host=self.host, port=self.port, db=self.db)
        logging.debug("Pinging the Redis server...")
        while True:
            return_value = self.client.ping()
            if return_value is True:
                logging.debug(
                    f"The Redis server responded with {return_value}!"
                )
                break
            else:
                logging.debug("The Redis server did not respond...")
            time.sleep(self.wait_time)


class RedisMessageExchangeWrapper(RedisMessageExchange):
    def __init__(
        self, namespace: str = "queue", *args: Any, **kwargs: Any
    ) -> None:
        """
        Initialize and set given class variables on class instantiation.

        Args:
            namespace (str, optional): The namespace used to form a unique key
                for the list/queue. The unique key is used to access the \
                list/queue. Defaults to "queue".
        """
        # Set the queue namespace
        self.namespace = namespace
        # Set the queue name
        self.name: Union[None, str] = None
        super().__init__(*args, **kwargs)


class InMemoryMessageExchangeWrapper:
    """A wrapper class to mimic the 'RedisMessageExchangeWrapper' class."""

    def __init__(self, *args: Tuple, **kwargs: Dict) -> None:
        """Initialize and set given class variables on class instantiation."""
        self.client = InMemoryMessageExchange()

    # Wrap kv store operations

    def kv_set(self, key: str, data: Any, *args: Tuple, **kwargs: Dict) -> bool:
        """
        SET data in memory.

        Args:
            data (Any): The data that should be SET in memory and that also \
                contains a unique identifier 'message_uuid' that can be used \
                to retrieve a possible result.
            key (str): The key that the data should be associated with.
        """
        return self.client.kv_set(key, data)

    def kv_get(
        self, key: str, *args: Tuple, **kwargs: Dict
    ) -> Union[None, Any]:
        """
        GET data from memory.

        Args:
            key (str): The key that the data is associated with.

        Returns:
            Any: The data retrieved from memory.
        """
        return self.client.kv_get(key)


class InMemoryMessageExchange:
    """A class that replicates Redis datastructures and operations."""

    def __init__(self):
        """Initialize and set given class variables on class instantiation."""
        self.qs: Dict[str, Any] = {}
        self.kvs: Dict[str, Any] = {}

    def delete(self, name: str) -> bool:
        """
        Delete the data associated with a certain name.

        Args:
            name (str): The name that is associated with certain data stored \
                in memory.

        Returns:
            int: 1 if the data was deleted successfully, otherwise 0.
        """
        if self._name_is_present(name=name, ds=QUEUE_IDENTIFIER):
            key = self._get_key(name, QUEUE_IDENTIFIER)
            # Remove deque object from dict
            del self.qs[key]
            return True
        elif self._name_is_present(name=name, ds=KVSTORE_IDENTIFIER):
            key = self._get_key(name, KVSTORE_IDENTIFIER)
            # Remove deque object from dict
            del self.kvs[key]
            return True
        else:
            return False

    def exists(self, name: str) -> bool:
        """
        Check if a certain key specified by a name exists.

        Args:
            name (str): The name that is associated with certain data stored \
                in memory.

        Returns:
            bool: True if the data exists, otherwise False.
        """
        if self._name_is_present(name=name, ds=QUEUE_IDENTIFIER):
            return True
        elif self._name_is_present(name=name, ds=KVSTORE_IDENTIFIER):
            return True
        else:
            return False

    def lrange(self, name: str, start: int, end: int) -> Union[None, List[Any]]:
        """
        Retrieve a slice of a list/queue.

        Args:
            name (str): The name of the list/queue to retrieve a list of \
                element from.
            start (int): Start index. Can be negative numbers just like Python \
                slicing notation.
            end (int): End index. Can be negative numbers just like Python \
                slicing notation.

        Returns:
            Union[None, List[Any]]: A slice of the list/queue 'name' between a \
                'start' and 'end' index.
        """
        key = self._get_key(name, QUEUE_IDENTIFIER)
        if self._name_is_present(name=name, ds=QUEUE_IDENTIFIER):
            start, end = self._adjust_indices(
                start=start, end=end, qlen=len(self.qs[key])
            )
            # Include right most value in range query
            return list(itertools.islice(self.qs[key], start, end + 1))  # noqa
        else:
            return None

    def llen(self, name: str) -> int:
        """
        Return the length of a list/queue.

        Args:
            name (str): The name of the list/queue.

        Returns:
            int: The length of the list/queue.
        """
        key = self._get_key(name, QUEUE_IDENTIFIER)
        if self._name_is_present(name=name, ds=QUEUE_IDENTIFIER):
            return len(self.qs[key])
        else:
            return 0

    def lindex(self, name: str, index: int) -> Union[None, Any]:
        """
        Return an element from a list/queue at a certain position.

        Args:
            name (str): The name of the list/queue.
            index (int): The index of the element in a list/queue to retrieve. \
                Negative indices are supported and will return an element \
                starting at the end of the list/queue.

        Returns:
            Union[None, Any]: An element from a list/queue at a certain \
                position.
        """
        key = self._get_key(name, QUEUE_IDENTIFIER)
        if self._name_is_present(name=name, ds=QUEUE_IDENTIFIER):
            try:
                return self.qs[key][index]
            except IndexError:
                return None
        else:
            return None

    def pipeline(self) -> "Pipeline":
        """
        Chain several operations on an in-memory datastructure.

        Returns:
            Pipeline: A pipeline object.
        """
        return Pipeline(queue=self)

    def rpush(self, name: str, value: Any) -> int:
        """
        Push an element onto the tail of the list/queue.

        Args:
            name (str): The name of the list/queue.
            value (Any): The element to add onto the tail of the \
                list/queue.

        Returns:
            int: The new number of elements in the list/queue.
        """
        key = self._get_key(name, QUEUE_IDENTIFIER)
        if self._name_is_present(name=name, ds=QUEUE_IDENTIFIER):
            # Extend right side of dequeue
            self.qs[key].extend([value])
            return len(self.qs[key])
        else:
            self.qs[key] = deque()
            # Extend right side of dequeue
            self.qs[key].extend([value])
            return len(self.qs[key])

    def rpop(self, name: str) -> Union[None, Any]:
        """
        Remove and return the last element of the list/queue.

        Args:
            name (str): The name of the list/queue.

        Returns:
            Union[None, Any]: The last element of the list/queue that was \
                removed fromt the list/queue, otherwise None.
        """
        key = self._get_key(name, QUEUE_IDENTIFIER)
        if self._name_is_present(name=name, ds=QUEUE_IDENTIFIER):
            # Remove element from right side of dequeue
            elem = self.qs[key].pop()
            if len(self.qs[key]) == 0:
                del self.qs[key]
            return elem
        else:
            return None

    def lpop(self, name: str) -> Union[None, Any]:
        """
        Remove and return the first element of the list/queue.

        Args:
            name (str): The name of the list/queue.

        Returns:
            Union[None, Any]: The first element of the list/queue that was \
                removed fromt the list/queue, otherwise None.
        """
        key = self._get_key(name, QUEUE_IDENTIFIER)
        if self._name_is_present(name=name, ds=QUEUE_IDENTIFIER):
            # Remove element from right side of dequeue
            elem = self.qs[key].popleft()
            if len(self.qs[key]) == 0:
                del self.qs[key]
            return elem
        else:
            return None

    def lset(self, name: str, index: int, value: Any,) -> Union[None, int]:
        """
        If possible, set an element at a certain position in the list/queue.

        Args:
            name (str): The name of the list/queue.
            index (int): The position in the list/queue that should be updated/\
                overwritten.
            value (Any): The data that should be set/stored at a \
                certain position in the list/queue.

        Returns:
            Union[None, int]:
        """
        key = self._get_key(name, QUEUE_IDENTIFIER)
        if self._name_is_present(name=name, ds=QUEUE_IDENTIFIER):
            try:
                self.qs[key][index] = value
                return 1
            except IndexError:
                return 0
        else:
            return None

    def kv_set(self, name: str, value: Any) -> bool:
        """
        Set an element in memory using a certain name/key.

        Args:
            name (str): The name a given element should be associated with \
                set under.
            value (Any): The element to set/store in memory.

        Returns:
            bool: True if the element was set/stored successfully, otherwise \
                False.
        """

        key = self._get_key(name, KVSTORE_IDENTIFIER)
        self.kvs[key] = value
        return True

    def kv_get(self, name: str) -> Union[None, Any]:
        """
        Return the element associated with a certain name.

        Args:
            name (str): The name/key associated with the element that is to be \
                retrieved from memory.

        Returns:
            Union[None, Any]: The element associated with the given name/key \
                or None if the name/key doesn't exist
        """
        key = self._get_key(name, KVSTORE_IDENTIFIER)
        if self._name_is_present(name=name, ds=KVSTORE_IDENTIFIER):
            return self.kvs[key]
        else:
            return None

    def _name_is_present(self, name: str, ds: str) -> bool:
        """
        Check if a certain name is present in an in-memory datastructure.

        Args:
            name (str): The name to check.
            ds (str): A specification of the in-memory datastructure to check.

        Raises:
            ValueError: If the specified datastructure does not exist.

        Returns:
            bool: True if the name is present, otherwise False.
        """
        key = self._get_key(name, ds)
        if ds.lower() == KVSTORE_IDENTIFIER:
            try:
                self.kvs[key]
                return True
            except KeyError:
                return False
        elif ds.lower() == QUEUE_IDENTIFIER:
            try:
                self.qs[key]
                return True
            except KeyError:
                return False
        else:
            raise ValueError(
                f"Specified datastructure: {ds} not known. Choose either \
                'kvstore' or 'queue'.",
            )

    def _adjust_indices(
        self, start: int, end: int, qlen: int
    ) -> Union[List, Tuple[int, int]]:
        """
        An internal method that is used to map negative indices to positive.

        This method is used to map negative indices to positive indices, so \
        that we can slice an iterator efficiently. It is not possible to slice \
        a 'collections.deque' object using negative indices. This method \
        implements this functionality.

        Args:
            start (int): Start index.
            end (int): End index.
            qlen (int): The length of a certain list/queue.

        Returns:
            Union[List, Tuple[int, int]]: Adjusted indices.
        """
        # -> Handle edge cases first
        if start >= qlen:
            return []
        if start <= -qlen:
            start = 0
        if end >= qlen:
            end = qlen
        if end < -qlen:
            return []
        # -> Then map valid negative indices to positive
        if start < 0:
            start += qlen
        if end < 0:
            end += qlen
        return start, end

    def _get_key(self, name: str, ds: str) -> str:
        """
        Retrieve an appropriate key for accessing an in-memory datastructure.

        Args:
            name (str): The name associated with certain data that is stored \
                in memory.
            ds (str): A specification of the in-memory datastructure to use.

        Returns:
            str: An appropriate key for accessing a certian in-memory \
                datastructure.
        """
        return f"{ds}:{name}"


class Pipeline:
    """Chain several operations on implemented datastructures."""

    def __init__(self, queue: InMemoryMessageExchange) -> None:
        self.queue = queue
        self.operations: List[Tuple[str, Any]] = []

    def delete(self, name: str) -> None:
        self.operations.append(("delete", (name,)))

    def execute(self) -> Any:
        rvs = []
        for operation, args in self.operations:
            rv = getattr(self.queue, operation)(*args)
            rvs.append(rv)
        # Reset list of operations to execute
        self.operations = []
        return rvs

    def lrange(self, name: str, start: int, end: int) -> None:
        self.operations.append(("lrange", (name, start, end)))

    def llen(self, name: str) -> None:
        self.operations.append(("llen", (name,)))

    def lindex(self, name: str, index: int) -> None:
        self.operations.append(("lindex", (name, index)))

    def rpush(self, name: str, value: Any) -> None:
        self.operations.append(("rpush", (name, value)))

    def rpop(self, name: str) -> None:
        self.operations.append(("rpop", (name,)))

    def lpop(self, name: str) -> None:
        self.operations.append(("lpop", (name,)))

    def lset(self, name: str, index: int, value: Any) -> None:
        self.operations.append(("lset", (name, index, value)))

    def kv_set(self, name: str, value: Any) -> None:
        self.operations.append(("kv_set", (name, value)))

    def kv_get(self, name: str) -> None:
        self.operations.append(("kv_get", (name,)))


def run_tests_in_memory_message_exchange():
    # 1: Test range queries
    queue = InMemoryMessageExchange()
    name = "KeyOne"
    ntimes = 100
    # Insert 100 integers (0,  ..., 99) into the queue with name 'KeyOne'
    queue = test_rpush(queue=queue, name=name, ntimes=ntimes)
    test_lrange(queue=queue, name=name, ntimes=ntimes)

    # 2: Test the rpop operation
    queue = InMemoryMessageExchange()
    name = "KeyOne"
    ntimes = 100
    # Insert 100 integers (0,  ..., 99) into the queue with name 'KeyOne'
    queue = test_rpush(queue=queue, name=name, ntimes=ntimes)
    test_rpop(queue=queue, name=name, ntimes=ntimes)

    # 3: Test the lpop operation
    queue = InMemoryMessageExchange()
    name = "KeyOne"
    ntimes = 100
    # Insert 100 integers (0,  ..., 99) into the queue with name 'KeyOne'
    queue = test_rpush(queue=queue, name=name, ntimes=ntimes)
    test_lpop(queue=queue, name=name, ntimes=ntimes)

    # 4: Test the lset operation
    queue = InMemoryMessageExchange()
    name = "KeyOne"
    ntimes = 100
    # Insert 100 integers (0,  ..., 99) into the queue with name 'KeyOne'
    queue = test_rpush(queue=queue, name=name, ntimes=ntimes)
    test_lset(queue=queue, name=name, ntimes=ntimes)

    # 5: Test kv_set, kv_get and delete
    queue = InMemoryMessageExchange()
    name = "KeyOne"
    ntimes = 100
    queue = test_kv_set(queue=queue, name=name, ntimes=ntimes)
    queue = test_kv_get(queue=queue, name=name, ntimes=ntimes)
    test_kv_delete(queue=queue, name=name, ntimes=ntimes)

    # 5: Test pipelining of queue operations
    test_pipeline()


def run_tests_redis_message_exchange():
    # TODO: Duplicate tests for Redis!
    print("  TODO: Duplicate tests for Redis!  ")


# Queue operations


def test_lrange(queue: InMemoryMessageExchange, name: str, ntimes: int) -> None:
    for index_i in range(-ntimes, ntimes):
        for index_j in range(-ntimes, ntimes):
            values = queue.lrange(name, start=index_i, end=index_j)
            key = queue._get_key(name=name, ds=QUEUE_IDENTIFIER)
            start, end = queue._adjust_indices(
                start=index_i, end=index_j, qlen=len(queue.qs[key])
            )
            if values is not None:
                if len(values) == 1:
                    assert values[0] == str(start)
                elif len(values) > 1:
                    assert values[0] == str(start)
                    assert values[-1] == str(end)


def test_rpush(
    queue: InMemoryMessageExchange, name: str, ntimes: int
) -> InMemoryMessageExchange:
    for i in range(ntimes):
        assert queue.rpush(name, str(i)) == (i + 1)
        assert queue.llen(name) == (i + 1)
        assert queue.lindex(name, i) == str(i)
    return queue


def test_rpop(queue: InMemoryMessageExchange, name: str, ntimes: int) -> None:
    for i in range(ntimes):
        value = queue.rpop(name)
        assert value == str(ntimes - (i + 1))
    # Check that the queue has been emptied
    assert queue.llen(name) == 0


def test_lpop(queue: InMemoryMessageExchange, name: str, ntimes: int) -> None:
    for i in range(ntimes):
        value = queue.lpop(name)
        assert value == str(i)
    # Check that the queue has been emptied
    assert queue.llen(name) == 0


def test_lset(queue: InMemoryMessageExchange, name: str, ntimes: int) -> None:
    # Set every other element to a different value
    for i in range(ntimes):
        if i % 2 == 0:
            queue.lset(name=name, index=i, value=str(-1))
    # Check that the change took effect...
    for i in range(ntimes):
        if i % 2 == 0:
            assert queue.lindex(name, i) == str(-1)
        else:
            assert queue.lindex(name, i) == str(i)


def test_kv_delete(
    queue: InMemoryMessageExchange, name: str, ntimes: int
) -> None:
    for i in range(ntimes):
        queue.delete(name + f"{i}")
    for i in range(ntimes):
        assert queue.kv_get(name + f"{i}") is None


# KV store operations


def test_kv_set(
    queue: InMemoryMessageExchange, name: str, ntimes: int,
) -> InMemoryMessageExchange:
    # Set a bunch of values
    for i in range(ntimes):
        queue.kv_set(name + f"{i}", str(i))
    return queue


def test_kv_get(
    queue: InMemoryMessageExchange, name: str, ntimes: int,
) -> InMemoryMessageExchange:
    for i in range(ntimes):
        assert queue.kv_get(name + f"{i}") == str(i)
    return queue


def test_pipeline():
    # Test the pipeline functionality
    queue = InMemoryMessageExchange()
    name = "KeyOne"
    ntimes = 100  # noqa
    # Set the number of operations we plan to execute
    noperations = 3
    # Create a pipeline object for executing a sequence
    # of operations in order
    pipe = Pipeline(queue)
    for i in range(ntimes):
        pipe.rpush(name, str(i))
        pipe.llen(name)
        pipe.lindex(name, i)
    # Get return values
    rvs = pipe.execute()
    # Check the data currently in the queue is correct
    for i in range(ntimes):
        assert queue.lindex(name, i) == str(i)
    # Check that the return values are correct
    for i in range(0, ntimes):
        rv0 = rvs[i * noperations + 0]
        rv1 = rvs[i * noperations + 1]
        rv2 = rvs[i * noperations + 2]
        # Value from rpush operation
        # --> it is the length of the queue after an rpush operation
        assert rv0 == i + 1
        # Value from llen operation
        # --> it is the length of the queue at this point
        assert rv1 == i + 1
        # Value from lindex operation
        # --> it is the value at a certain index
        assert rv2 == str(i)


if __name__ == "__main__":
    #
    print("\nRunning 'InMemoryMessageExchange' tests...\n")
    run_tests_in_memory_message_exchange()
    print("\nDone...\n")
    #
    print("\nRunning 'RedisMessageExchange' tests...\n")
    run_tests_redis_message_exchange()
    print("\nDone...\n")
