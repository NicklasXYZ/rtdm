import os
from datetime import datetime, timedelta
from typing import List

import django
import msgpack
import ruuid

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stream_handler.settings")
django.setup()

import backend.models as django_models
from django.conf import settings
import backend.utils as utils
from backend.DataStructures import BaseDataPoint, DataPoint, Trajectory

TEST_USER_UUID = str(ruuid.uuid4())
TEST_START_TIMESTAMP = datetime(2021, 8, 26, 12, 4, 24, 1000)


# BaseDataPoint tests


def test_base_data_point():
    bdp0 = BaseDataPoint(
        latitude=settings.PRIME_MERIDIAN_COORDS[0],
        longitude=settings.PRIME_MERIDIAN_COORDS[1],
        external_timestamp=datetime(2021, 8, 26, 12, 4, 24, 1000),
        user=TEST_USER_UUID,
    )

    # Test: Geohashing functionality
    assert (
        bdp0.get_geohashed_datapoint(precision=16, bits_per_char=2,).geohash
        == "1233233233223003"
    )
    assert (
        bdp0.get_geohashed_datapoint(precision=17, bits_per_char=2,).geohash
        == "12332332332230032"
    )
    assert (
        bdp0.get_geohashed_datapoint(precision=18, bits_per_char=2,).geohash
        == "123323323322300320"
    )
    print(bdp0.get_unix_timestamp())
    assert bdp0.get_unix_timestamp() in [1629975864.001, None]

    # Test: Given timestamp is None. Unix timestamp is thus also None
    bdp1 = BaseDataPoint(
        latitude=settings.PRIME_MERIDIAN_COORDS[0],
        longitude=settings.PRIME_MERIDIAN_COORDS[1],
        external_timestamp=None,
        user=TEST_USER_UUID,
    )
    assert bdp1.get_unix_timestamp() == None


# DataPoint tests


def _assert_datapoint_dict(dp, dp_dict):
    assert dp_dict["latitude"] == dp.latitude
    assert dp_dict["longitude"] == dp.longitude
    assert dp_dict["external_timestamp"] == str(dp.external_timestamp)
    assert dp_dict["dx"] == dp.dx
    assert dp_dict["dt"] == dp.dt
    assert dp_dict["accuracy"] == dp.accuracy
    assert dp_dict["speed"] == dp.speed
    assert dp_dict["acceleration"] == dp.acceleration


def _assert_datapoint_model(dp, dp_model):
    assert dp_model.latitude == dp.latitude
    assert dp_model.longitude == dp.longitude
    assert dp_model.external_timestamp == dp.external_timestamp
    assert dp_model.dx == dp.dx
    assert dp_model.dt == dp.dt
    assert dp_model.accuracy == dp.accuracy
    assert dp_model.speed == dp.speed
    assert dp_model.acceleration == dp.acceleration


def test_datapoint():
    dp0 = DataPoint(
        latitude=settings.PRIME_MERIDIAN_COORDS[0],
        longitude=settings.PRIME_MERIDIAN_COORDS[1],
        external_timestamp=TEST_START_TIMESTAMP,
        user=TEST_USER_UUID,
    )
    dp_list = [
        DataPoint(
            latitude=settings.PRIME_MERIDIAN_COORDS[0] + i / 1000,
            longitude=settings.PRIME_MERIDIAN_COORDS[1] + i / 1000,
            external_timestamp=TEST_START_TIMESTAMP + timedelta(seconds=i),
            user=TEST_USER_UUID,
        )
        for i in range(10)
    ]
    dp0.datapoints = dp_list
    _test_datapoint_conversions(dp0=dp0, dp_list=dp_list)


def _test_datapoint_conversions(
    dp0: DataPoint, dp_list: List[DataPoint],
) -> None:
    ###
    # Test: Conversion between Python object and dictionary
    dp0_dict = dp0.to_dict()
    # DataPoint --> Dict:
    #   - Make sure fields are available and correctly set
    if isinstance(dp0_dict, dict):
        assert True
        _assert_datapoint_dict(dp0, dp0_dict)
        assert DataPoint.from_dict(dp0_dict) == dp0
        datapoints = dp0_dict["datapoints"]
        for i in range(len(datapoints)):
            _assert_datapoint_dict(dp_list[i], datapoints[i])
            assert DataPoint.from_dict(datapoints[i]) == dp_list[i]
    else:
        assert False

    ###
    # Test: Conversion between Python object and Django database model
    dp0_django_model = dp0.to_model()
    # DataPoint --> Model:
    #   - Make sure fields are available and correctly set
    if isinstance(dp0_django_model, django_models.DataPoint):
        assert True
        _assert_datapoint_model(dp0, dp0_django_model)
        assert DataPoint.from_model(dp0_django_model) == dp0
        datapoints = msgpack.loads(dp0_django_model.datapoints)
        for i in range(len(datapoints)):
            _assert_datapoint_dict(dp_list[i], datapoints[i])
            assert DataPoint.from_dict(datapoints[i]) == dp_list[i]
    else:
        assert False


# Trajectory tests


def _assert_trajectory_dict(t, t_dict):
    # assert t_dict["uid"] == t.uid
    # assert t_dict["trajectory"] == t.trajectory
    # assert t_dict["color"] == t.color
    assert t_dict["user"] == t.user
    assert t_dict["tag"] == t.tag


def _assert_trajectory_model(t, t_model):
    # assert t_model.uid == t.uid
    assert t_model.trajectory == t.trajectory
    # assert t_model.color == t.color
    assert t_model.user == t.user
    assert t_model.tag == t.tag


def test_trajectory():
    t0 = Trajectory(
        datapoints=[],
        tag="raw",
        user=TEST_USER_UUID,
        color=f"#{utils.color_hex()}",
        # start_timestamp = None,
        # end_timestamp = None,
    )
    # Generate a sequence of points
    n = 10
    dps = []
    dp_lists = []
    time = 0
    for _ in range(n):
        dp0 = DataPoint(
            latitude=settings.PRIME_MERIDIAN_COORDS[0],
            longitude=settings.PRIME_MERIDIAN_COORDS[1],
            external_timestamp=TEST_START_TIMESTAMP + timedelta(seconds=time),
            user=TEST_USER_UUID,
        )
        # Generate nested points
        dp_list = [
            DataPoint(
                latitude=settings.PRIME_MERIDIAN_COORDS[0]
                + (j + time) / 1000,
                longitude=settings.PRIME_MERIDIAN_COORDS[1]
                + (j + time) / 1000,
                external_timestamp=TEST_START_TIMESTAMP
                + timedelta(seconds=j + time),
                user=TEST_USER_UUID,
            )
            for j in range(n)
        ]
        dp0.datapoints = dp_list
        # Cumulate time gap
        time += n + 1
        dps.append(dp0)
        dp_lists.append(dp_list)
    t0.datapoints = dps
    _test_trajectory_conversions(t0=t0, dp_lists=dp_lists)


def _test_trajectory_conversions(
    t0: Trajectory, dp_lists=List[List[DataPoint]],
) -> None:
    ###
    # Test: Conversion between Python object and dictionary
    t0_dict = t0.to_dict()
    # DataPoint --> Dict:
    #   - Make sure fields are available and correctly set
    if isinstance(t0_dict, dict):
        assert True
        _assert_trajectory_dict(t0, t0_dict)
        assert Trajectory.from_dict(t0_dict) == t0
        for i in range(len(t0.datapoints)):
            _assert_datapoint_dict(t0.datapoints[i], t0_dict["datapoints"][i])
            assert (
                DataPoint.from_dict(t0_dict["datapoints"][i])
                == t0.datapoints[i]
            )
    else:
        assert False

    ###
    # Test: Conversion between Python object and Django database model
    t0_django_model = t0.to_model()
    # DataPoint --> Model:
    #   - Make sure fields are available and correctly set
    if isinstance(t0_django_model, django_models.Trajectory):
        assert True
        _assert_trajectory_model(t0, t0_django_model)
        assert Trajectory.from_dict(t0_dict) == t0
        # for i in range(len(t0.datapoints)):
        #     _assert_datapoint_model(t0.datapoints[i], t0_django_model["datapoints"][i])
        #     assert DataPoint.from_model(t0_django_model["datapoints"][i]) == t0.datapoints[i]
    else:
        assert False
