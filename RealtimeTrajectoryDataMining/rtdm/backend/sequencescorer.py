import collections
import logging
import random
import string
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Tuple, Union

import backend.models as django_models
import backend.xchanges as xchanges
import msgpack  # pip install msgpack
import numpy as np  # pip install numpy
import redis  # pip install redis-py
from backend.typealias import Pattern, Sequence, Token
from Bio import pairwise2  # pip install biopython
from prefixspan import PrefixSpan  # pip install prefixspan


class ScoringFunction:
    def __init__(self, fn: Callable, *fn_args: Any, **fn_kwargs: Any) -> None:
        """
        Initialize and set given class variables on class instantiation.

        Args:
            fn (Union[None, Callable], optional): A scoring function that takes
                two lists of tokens and returns an alignmnet score. Defaults to
                None.
        """
        self.fn = fn
        self.fn_args = fn_args
        self.fn_kwargs = fn_kwargs
        # Set defaults for optional args:
        # - NOTE: 'gap_char' required for list input which is used by default
        #   in this script
        self.fn_kwargs.setdefault("gap_char", ["-"])
        self.fn_kwargs.setdefault("score_only", True)
        # Make sure the function can actually be evaluated using expected input
        # and that the expected output is produced
        self._is_valid()

    def _is_valid(self):
        """
        Check that the user-defined callable/function is valid and works.

        Raises:
            TypeError: If given callable/function does produce expected output.
            Exception: If given callable/function does not work with expected \
                input.
        """
        try:
            output = self._test_fn()
            if not isinstance(output, float):
                raise TypeError(
                    f"The output value of the scoring function is not valid! \
                    It needs to be of type 'float' but {type(output)} was \
                    produced!"
                )
        except Exception as e:
            raise Exception(
                f"The given function can not be used as a valid scoring \
                function. The following exception was raised: {e}"
            )

    def _test_fn(self) -> float:
        """
        Test that the given callable/function works.

        Returns:
            float: An alignmnet score given two sequences.
        """
        test_support = test_generate_data(
            ntimes=5,
            min_seq_size=6,
            max_seq_size=8,
            lowercase_letters=False,
            replicate=4.00,
        )
        sequence0 = test_support[int(np.random.randint(0, len(test_support)))]
        sequence1 = test_support[int(np.random.randint(0, len(test_support)))]
        text0 = [str(i) for i in sequence0]
        text1 = [str(i) for i in sequence1]
        score = self.fn(text0, text1, *self.fn_args, **self.fn_kwargs)  # noqa
        return score


class BaseSequenceScorer:
    def __init__(
        self,
        exchange: Union[
            xchanges.InMemoryMessageExchangeWrapper,
            xchanges.RedisMessageExchangeWrapper,
        ],
        fns: Union[None, Dict[Any, Tuple[ScoringFunction, float]]] = None,
    ) -> None:
        """
        Initialize and set given class variables on class instantiation.

        Args:
            fns (Union[None, Dict[Any, Tuple[ScoringFunction, float]]], \
                optional): A dictionary of tuples each containing a scoring \
                function and corresponding weight. Defaults to None.
        """
        self._exchange = exchange
        self.fns = fns
        if fns is None:
            # Only use local sequence alignmnet algorithm
            f1 = ScoringFunction(pairwise2.align.localxx)

            # Only use global sequence alignmnet algorithm
            # f1 = sequencescorer.ScoringFunction(pairwise2.align.globalxx)

            self.fns = {"f1": (f1, 1.00)}

            # The local alignment function:
            # - local: Local alignment
            # - m    : A match score is the score of identical chars, otherwise
            #          mismatch score
            # - s    : Same open and extend gap penalties for both sequences
            # args = (1.00, -1.00, -1.00, -1.00)
            # f1 = ScoringFunction(pairwise2.align.localms, *args)

            # The global alignmnet function:
            # - global: Global alignment
            # - m    : A match score is the score of identical chars, otherwise
            #          mismatch score
            # - s    : Same open and extend gap penalties for both sequences
            # args = (1.00, -1.00, -1.00, -1.00)
            # f2 = ScoringFunction(pairwise2.align.globalms, *args)
            # Give most weight to the global alignmnet function
            # self.fns = {"f1": (f1, 0.25), "f2": (f2, 0.75)}

    def extract_support(
        self,
        database: List[Sequence],
        min_frequency: int = 1,
        closed: bool = True,
        generator: bool = False,
    ) -> List[Pattern]:
        """
        Extract frequent sequential patterns from a list of sequences.

        Args:
            database (List[Sequence]): A list of sequences (lists) made up of \
                tokens.
            min_frequency (int, optional): The minimum number of times a \
                pattern should appear before it is actually considered \
                    a pattern. Defaults to 1.
            closed (bool, optional): A pattern is closed if there is no \
                super-pattern with the same frequency. Defaults to True.
            generator (bool, optional): A pattern is generator if there is \
                no sub-pattern with the same frequency. Defaults to False.

        Raises:
            ValueError: If 'min_frequency' >= size of 'database'.
            ValueError: If no support (frequent sequential patterns) are found.

        Returns:
            List[Pattern]: The support set. A list of frequent \
                sequential patterns.
        """
        if len(database) < min_frequency:
            logging.warning(
                f"Support set could be extracted. The number of available \
                sequences ({len(database)}) need to be >= {min_frequency}"
            )
            raise ValueError(
                f"The number of available sequences needs to be >= \
                {min_frequency}"
            )
        # Use algorithm 'PrefixSpan' to discover sequence patterns that are
        # common across sequence data
        ps = PrefixSpan(database)
        # These are the sequences we are interested in:
        # - Sequences where no super-pattern exist with the same frequency > 2
        # --> This is the case for default values: closed = True and
        # generator = False
        coverage: List[List[Any]] = [[] for i in range(len(database))]

        # Callback function to prune redundant sequence patterns
        def cover(patt, matches):
            for i, _ in matches:
                coverage[i] = max(coverage[i], patt, key=len)

        ps.frequent(
            minsup=min_frequency,
            closed=closed,
            generator=generator,
            callback=cover,
        )
        sequences_ = collections.Counter(tuple(x) for x in iter(coverage))
        sequences = [(j, list(i)) for i, j in sequences_.items()]
        if len(sequences) == 0:
            logging.warning(
                f"No frequent sequential patterns were found by PrefixSpan! \
                The number of available sequences is: ({len(database)}) \
                The minimum frequency is: {min_frequency}"
            )
            # Calling function should handle this error:
            raise ValueError("No frequent sequential patterns were found!")
        else:
            return sequences

    # TODO: Check if this function can be unified with method in file 'utils.py'
    def get_key(self, obj_type: str, uid: str) -> str:
        """
        Construct a proper key for accessing a particular value in Redis.

        Args:
            obj_type (str): Object type/category used as a namespace.
            uid (str): A unique identifier of an entity.

        Returns:
            str: A key to a value in Redis.
        """
        return f"{obj_type}:{uid}"

    def get_support(
        self, uid: str, obj_type: str = "support",
    ) -> Union[None, List[Pattern]]:
        """
        Retrieve the set of frequent sequential patterns belonging to a user.

        Args:
            uid (str): The unique identifier of an entity.
            obj_type (str, optional): A namespace used to categorize different \
                types of data inserted into Redis. Defaults to the string: \
                "support".

        Returns:
            Union[None, List[Pattern]]: Return requested support set if \
                possible, otherwise None.
        """
        if not isinstance(uid, str):
            raise TypeError("Expected input argument 'uid' to be a string")
        if not isinstance(obj_type, str):
            raise TypeError("Expected input argument 'obj_type' to be a string")
        # Contruct the key of the value in Redis that we want to access
        key = self.get_key(obj_type=obj_type, uid=uid)
        support = self._get_support(key=key, uid=uid)
        return support

    def score_sequence(
        self, uid: str, sequence: List[Token],  # TODO: Update type 'Sequence'
    ) -> Union[None, float]:
        """
        Compute a similarity score.

        Note:
            Method used for scoring a sequence given the unique identifier of \
            a user. The support set, i.e. a set of frequent sequential \
            patterns, is retrieved based on the given unique identifier. Based \
            on the support set an anomaly score is calculated and returned.

        Args:
            uid (str): The unique identifier of an entity.
            sequence (List[Token]): A sequence that is to be scored.

        Raises:
            ValueError: If a wrongly formatted input sequence is given.

        Returns:
            float: The anomaly score computed as a weighted average of the \
                alignment scores between the new sequence and the sequences \
                in the support set.
        """
        support = self.get_support(uid=uid)
        if support is not None:
            if not isinstance(sequence, list):
                logging.warning(
                    f"Wrong input given. Expected input 'sequence' to be a \
                    list of tokens, but {type(sequence)} was given."
                )
                raise TypeError(
                    "Expected input 'sequence' to be a list of tokens!"
                )
            else:
                return self._score_sequence(support=support, sequence=sequence)
        else:
            return None

    def clear(self, uid: str, obj_type: str = "support",) -> bool:
        """
        Clear all sequence data associated with a certain user.

        Args:
            uid (str): The unique identifier of an entity.
            obj_type (str, optional): A namespace used to categorize \
                different types of data inserted into the Redis KV store. \
                Defaults to the string: "support".

        Returns:
            bool: True if successful, otherwise False.
        """
        if not isinstance(uid, str):
            raise TypeError("Expected input argument 'uid' to be a string")
        if not isinstance(obj_type, str):
            raise TypeError("Expected input argument 'obj_type' to be a string")
        # Contruct the key of the value in Redis that we want to access
        key = self.get_key(obj_type=obj_type, uid=uid)
        # Delete cached support set associated with the entity
        # TODO: Handle re-connection logic / print error message of failed
        # attempt
        return self._clear(key=key, uid=uid)

    def _get_support(self, key: str, uid: str) -> Union[None, List[Pattern]]:
        # This method should be implemented in a subclass
        raise NotImplementedError("Missing implementation in subclass!")

    def _score_sequence(
        self, support: List[Pattern], sequence: List[Token],
    ) -> Union[None, float]:
        """
        Compute an anomaly score of a given sequence.

        Args:
            support (List[Pattern]): A list of frequent sequential patterns.
            sequence (List[Token]): A sequence that is to be scored.

        Returns:
            float: An anomaly score associated with the given input sequence.
        """
        # This method should be implemented in a subclass
        raise NotImplementedError("Missing implementation in subclass!")

    def _create_support(
        self,
        uid: str,
        sequences: List[Sequence],
        min_frequency: int = 1,
        closed: bool = True,
        generator: bool = False,
    ) -> List[Pattern]:
        # This method should be implemented in a subclass
        raise NotImplementedError("Missing implementation in subclass!")

    def _clear(self, key: str, uid: str) -> bool:
        # This method should be implemented in a subclass
        raise NotImplementedError("Missing implementation in subclass!")


class InMemorySequenceScorer(BaseSequenceScorer):
    def __init__(self, **kwargs: Any) -> None:
        # Use a dictionary of ordered dictionaries for data storage
        # --> Each entry in the dictionary '_dicts' is associated with a user
        # --> Each entry is an ordered dictionary
        # --> An ordered dictionary will contain a 'support set'
        self._dicts: Dict[str, Any] = {}
        super().__init__(**kwargs)

    def _create_support(
        self,
        uid: str,
        sequences: List[Sequence],
        min_frequency: int = 1,
        closed: bool = True,
        generator: bool = False,
    ) -> List[Pattern]:
        support = self.extract_support(
            sequences,
            min_frequency=min_frequency,
            closed=closed,
            generator=generator,
        )
        self._dicts[uid] = support
        # self._add_support(
        #     key = key,
        #     support = msgpack.dumps(support),
        # )
        return support

    # TODO: Possibly rename...
    def _get_or_create_support(
        self, key: str, uid: str
    ) -> Union[None, List[Pattern]]:
        """
        Get or create the support set.

        Args:
            uid (str): A unique identifier of an entity.

        Returns:
            Union[None, List[Pattern]]: Return requested support set if \
                possible, otherwise None.
        """
        value = self._exchange.kv_get(key=key)
        if value is not None:
            return msgpack.loads(value)
        else:
            try:
                return self._dicts[uid]
            except KeyError:
                return None
                # TODO: Should the support be generated here?
                # support = self._create_support(uid = uid)
                # return support

    def _add_support(self, key: str, support: bytes) -> bool:
        """
        Add a frequent sequential patterns to Redis.

        Args:
            key (str): The key to the value in Redis.
            support (bytes): The encoded support set that should be added to
                Redis for easy and effecient access.

        Returns:
            bool: A status code indicating whether the attempt to insert the \
                data was successful or not: 1 successful, otherwise 0.
        """
        return self._exchange.kv_set(key=key, data=support)

    def _get_support(self, key: str, uid: str,) -> Union[None, List[Pattern]]:
        """
        Retrieve the support set of frequent sequential patterns of a user.

        Args:
            key (str): The key used to specify the value in Redis that should \
                be retrieved. The key is made up of a namespace and a unique \
                identifier of an entity. An appropriate key can be generated \
                with the 'get_key()' method.

        Returns:
            Union[None, List[Pattern]]: Return requested support set if \
                possible, otherwise None.
        """
        if isinstance(self._exchange, xchanges.InMemoryMessageExchangeWrapper):
            if isinstance(
                self._exchange.client, xchanges.InMemoryMessageExchange
            ):
                # Check if the value that contains the support set exists in
                # cache
                if self._exchange.client.exists(key) == 0:
                    # -> If the value does not exist then retrieve the data
                    # from the database and cache the data by loading it into
                    # Redis
                    logging.debug(
                        f"Function: '{self._get_support.__name__}'. \
                        Information: Retriving information from the database!"
                    )
                    support = self._get_or_create_support(uid=uid, key=key)
                    if support is not None:
                        self._add_support(
                            key=key, support=msgpack.dumps(support),
                        )
                        return support
                    else:
                        logging.warning(
                            f"Function: '{self._get_support.__name__}'. \
                            Problem: The response was None! Could not retrieve \
                            the suppport set!"
                        )
                        return None
                # -> Otherwise, retrieve the existing data from Redis
                else:
                    logging.debug(
                        f"Function: '{self.get_support.__name__}'. \
                        Information: Reading cached data from Redis!"
                    )
                    # support = self._exchange.client.get(name = key)
                    support = self._exchange.kv_get(key=key)
                    return msgpack.loads(support)
            else:
                logging.warning("")
                raise ValueError("")
        else:
            logging.warning("")
            raise ValueError("")

    def _clear(self, key: str, uid: str) -> bool:
        """
        Clear all sequence data associated with a user.

        Args:
            uid (str): The unique identifier of an entity.
            obj_type (str, optional): A namespace used to categorize different \
                types of data inserted into Redis. Defaults to the string: \
                "support".

        Returns:
            bool: True if successful otherwise False.
        """
        if isinstance(self._exchange, xchanges.InMemoryMessageExchangeWrapper):
            if isinstance(
                self._exchange.client, xchanges.InMemoryMessageExchange
            ):
                return self._exchange.client.delete(key)
            else:
                logging.warning("")
                raise ValueError("")
        else:
            logging.warning("")
            raise ValueError("")


class SequenceScorer(BaseSequenceScorer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _add_support(
        self, key: str, support: bytes, ttl: Union[None, int] = None,
    ) -> bool:
        """
        Add frequent sequential patterns to Redis.

        Args:
            key (str): The key to the value in Redis.
            support (bytes): The encoded support set that should be added to \
                Redis for easy and effecient access.

        Returns:
            int: A status code indicating whether the attempt to insert the \
                data was successful or not: 1 successful, otherwise 0.
        """
        if ttl is not None and ttl is not isinstance(ttl, int):
            error = (
                f"ARG 'ttl' is of type {type(ttl)} but should be of type 'int'!"
            )
            try:
                ttl = int(ttl)
            except Exception:
                raise TypeError(error)
        if isinstance(self._exchange, xchanges.RedisMessageExchangeWrapper):
            if isinstance(self._exchange.client, redis.Redis):
                start_time = datetime.utcnow() + timedelta(
                    seconds=self._exchange.timeout
                )
                support_added = False
                while True:
                    if datetime.utcnow() - start_time > timedelta(
                        seconds=self._exchange.timeout
                    ):
                        logging.warning(
                            f"Waited {self._exchange.timeout} seconds. Data \
                            could not be added!"
                        )
                        return False  # Return status code 0
                    try:
                        if support_added is False:
                            return_code = self._exchange.client.set(
                                key, support, ex=ttl,
                            )
                            if return_code is True:
                                support_added = True
                        if support_added is True:
                            break
                    except redis.exceptions.ConnectionError:
                        # Try to fix the connection
                        self._exchange.reset_connection()
                    time.sleep(self._exchange.wait_time)
                return True  # Return status code 1: Successful
            else:
                logging.warning("")
                raise ValueError("")
        else:
            logging.warning("")
            raise ValueError("")

    def _get_support(self, key: str, uid: str,) -> Union[None, List[Pattern]]:
        """
        Retrieve the set of frequent sequential patterns belonging to a user.

        Args:
            key (str): The key used to specify the value in Redis that should \
                be retrieved. The key is made up of a namespace and a unique \
                identifier of an entity. An appropriate key can be generated \
                with the 'get_key()' method.

        Returns:
            Union[None, List[Pattern]]: Return requested support set if \
                possible, otherwise None.
        """
        if isinstance(self._exchange, xchanges.RedisMessageExchangeWrapper):
            if isinstance(self._exchange.client, redis.Redis):

                # Check if the value that contains the support set exists in
                # cache
                if self._exchange.client.exists(key) == 0:
                    # -> If the value does not exist then retrieve the data
                    # from the database and cache the data by loading it into
                    # Redis
                    logging.debug(
                        f"Function: '{self._get_support.__name__}'. \
                        Information: Retriving information from the database!"
                    )
                    support = self._get_or_create_support(uid)
                    if support is not None:
                        self._add_support(
                            key=key, support=msgpack.dumps(support),
                        )
                        return support
                    else:
                        logging.warning(
                            f"Function: '{self._get_support.__name__}'. \
                            Problem: The response was None! Could not retrieve \
                            the suppport set!"
                        )
                        return None
                # -> Otherwise, retrieve the existing data from Redis
                else:
                    logging.debug(
                        f"Function: '{self.get_support.__name__}'. \
                        Information: Reading cached data from Redis!"
                    )
                    support = self._exchange.client.get(name=key)
                    return msgpack.loads(support)
            else:
                logging.warning("")
                raise ValueError("")
        else:
            logging.warning("")
            raise ValueError("")

    def _create_support(
        self,
        uid: str,
        sequences: List[Sequence],
        min_frequency: int = 1,
        closed: bool = True,
        generator: bool = False,
    ) -> List[Pattern]:
        support = self.extract_support(sequences, min_frequency=min_frequency)
        support = django_models.SupportSet.objects.create(
            user=uid, patterns=support,
        )
        return support

    # TODO: Possibly rename...
    def _get_or_create_support(self, uid: str) -> Union[None, List[Pattern]]:
        """
        Get or create the support set.

        Args:
            uid (str): A unique identifier of an entity.

        Returns:
            Union[None, List[Pattern]]: Return requested support set if \
                possible, otherwise None.
        """
        try:
            support = django_models.SupportSet.objects.get(user=uid)
            return support.patterns
        except django_models.SupportSet.DoesNotExist:
            return None

    def _clear(self, key: str, uid: str,) -> bool:
        """
        Clear all sequence data associated with a user.

        Args:
            uid (str): The unique identifier of an entity.
            obj_type (str, optional): A namespace used to categorize different \
                types of data inserted into Redis. Defaults to the string: \
                "support".

        Returns:
            int: 1 if successful.
        """
        if isinstance(self._exchange, xchanges.RedisMessageExchangeWrapper):
            if isinstance(self._exchange.client, redis.Redis):
                return_value = self._exchange.client.delete(key)
                # Delete the support set associated with the entity from the
                # database
                # TODO: Handle failed attempt / print error message of failed
                # attempt
                django_models.SupportSet.objects.filter(uid=uid).delete()
                return return_value
            else:
                logging.warning("")
                raise ValueError("")
        else:
            logging.warning("")
            raise ValueError("")


class ExtendedInMemorySequenceScorer(InMemorySequenceScorer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    # Override the internal sequence scorer method
    def _score_sequence(
        self, support: List[Pattern], sequence: List[Token],
    ) -> Union[None, float]:
        if self.fns is not None:
            return _score_sequence(
                support=support, sequence=sequence, fns=self.fns,
            )
        else:
            return None


class ExtendedSequenceScorer(BaseSequenceScorer):

    # Override the internal sequence scorer method
    def _score_sequence(
        self, support: List[Pattern], sequence: List[Token],
    ) -> Union[None, float]:
        if self.fns is not None:
            return _score_sequence(
                support=support, sequence=sequence, fns=self.fns,
            )
        else:
            return None


def _score_sequence(
    support: List[Pattern],
    sequence: List[Token],
    fns: Dict[str, Tuple[ScoringFunction, float]],
) -> float:
    best_score = 0.0
    for _, support_seq in support:
        if len(support_seq) > 0:
            score = 0.0
            for fn, weight in fns.values():
                score += (
                    weight
                    * fn.fn(  # noqa
                        sequence, support_seq, *fn.fn_args, **fn.fn_kwargs,
                    )
                    / len(sequence)
                )
            if score > best_score:
                best_score = score
                # If we find a sequence that mathes perfectly, then we do not
                # need to compare against the remaining sequences
                if best_score == 1.0:
                    break
    return best_score


# Function for generating random test data
def test_generate_data(
    ntimes: int,
    min_seq_size: int,
    max_seq_size: int,
    digits: bool = True,
    uppercase_letters: bool = True,
    lowercase_letters: bool = True,
    replicate: float = 0.25,
) -> List[Sequence]:
    if min_seq_size > max_seq_size:
        raise ValueError("Input 'min_seq_size' is larger than 'max_seq_size'!")
    uppercase_letters_ = []
    lowercase_letters_ = []
    digits_ = []
    # Allow uppercase letters to appear in the randomly generated sequences
    if uppercase_letters:
        uppercase_letters_ = list(string.ascii_uppercase)
    # Allow lowercase letters to appear in the randomly generated sequences
    if lowercase_letters:
        lowercase_letters_ = list(string.ascii_lowercase)
    # Allow digits to appear in the randomly generated sequences
    if digits:
        digits_ = list(string.digits)
    values = uppercase_letters_ + lowercase_letters_ + digits_
    rand_seqs: List[Sequence] = []
    # Generate 'ntimes' random sequences
    for _ in range(ntimes):
        seq = random.choices(  # noqa
            values, k=random.randint(min_seq_size, max_seq_size)  # noqa
        )
        rand_seqs.append(seq)
    # Replicate a given percentage of the generated random sequences
    mtimes = int(len(rand_seqs) * replicate)
    data = []
    for _ in range(mtimes):
        rnd = random.randint(0, len(rand_seqs) - 1)  # noqa
        data.append(rand_seqs[rnd])
    rand_seqs.extend(data)
    return rand_seqs


# Function for creating a test user and associating random test data
# by inserting the data into the database on behalf of the user
def test_insert_data(rand_seqs: List[Sequence]) -> None:
    # Create a test user who the generated sequences are going to belong to
    try:
        print("Retriving the user")
        usr_obj = django_models.User.objects.get(username="Tester0")
    except django_models.User.DoesNotExist:
        print("The user already exists. A new user is created.")
        usr_obj = django_models.User.objects.create_user(
            username="Tester0", password="passw0rd!"
        )
    seq_objs = []
    for seq in rand_seqs:
        seq_objs.append(django_models.Sequence(tokens=seq, user=usr_obj))
    django_models.Sequence.objects.bulk_create(seq_objs)


def test_delete_data():
    # Delete all users and their related data
    django_models.User.objects.all().delete()


if __name__ == "__main__":
    rand_seqs = test_generate_data(
        ntimes=50,
        min_seq_size=4,
        max_seq_size=8,
        lowercase_letters=False,
        replicate=2.00,
    )
    test_insert_data(rand_seqs=rand_seqs)
