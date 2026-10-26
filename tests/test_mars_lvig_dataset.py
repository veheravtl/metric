"""Интеграционные тесты извлечения MARS-LVIG из ROS1 bag и ROS2 MCAP."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from rosbags.rosbag1 import Writer
from rosbags.rosbag2 import StoragePlugin
from rosbags.rosbag2 import Writer as Writer2
from rosbags.typesys import Stores, get_typestore

from aerial_mapper.mars_lvig_dataset import export_mars_lvig_bag

CAMERA_TOPIC = "/left_camera/image/compressed"
RTK_TOPIC = "/dji_osdk_ros/rtk_position"


def _header(typestore: object, timestamp_nanoseconds: int, sequence: int) -> object:
    """Создаёт ROS1 Header с явно контролируемой временной меткой."""

    time_type = typestore.types["builtin_interfaces/msg/Time"]
    header_type = typestore.types["std_msgs/msg/Header"]
    seconds, nanoseconds = divmod(timestamp_nanoseconds, 1_000_000_000)
    return header_type(
        seq=sequence,
        stamp=time_type(sec=seconds, nanosec=nanoseconds),
        frame_id="test",
    )


def _write_synthetic_bag(path: Path, *, include_rtk: bool = True) -> bytes:
    """Пишет два JPEG-кадра и две RTK-точки без зависимости от ROS runtime."""

    typestore = get_typestore(Stores.ROS1_NOETIC)
    compressed_type = typestore.types["sensor_msgs/msg/CompressedImage"]
    fix_type = typestore.types["sensor_msgs/msg/NavSatFix"]
    status_type = typestore.types["sensor_msgs/msg/NavSatStatus"]
    jpeg_payload = b"\xff\xd8\xffsynthetic-jpeg\xff\xd9"

    with Writer(path) as writer:
        image_connection = writer.add_connection(
            CAMERA_TOPIC,
            "sensor_msgs/msg/CompressedImage",
            typestore=typestore,
        )
        rtk_connection = None
        if include_rtk:
            rtk_connection = writer.add_connection(
                RTK_TOPIC,
                "sensor_msgs/msg/NavSatFix",
                typestore=typestore,
            )

        if rtk_connection is not None:
            first_fix = fix_type(
                header=_header(typestore, 1_000_000_000, 0),
                status=status_type(status=0, service=1),
                latitude=22.30,
                longitude=113.90,
                altitude=80.0,
                position_covariance=np.zeros(9, dtype=np.float64),
                position_covariance_type=0,
            )
            writer.write(
                rtk_connection,
                1_000_000_000,
                typestore.serialize_ros1(first_fix, first_fix.__msgtype__),
            )

        first_image = compressed_type(
            header=_header(typestore, 1_100_000_000, 0),
            format="jpeg",
            data=np.frombuffer(jpeg_payload, dtype=np.uint8).copy(),
        )
        writer.write(
            image_connection,
            1_100_000_000,
            typestore.serialize_ros1(first_image, first_image.__msgtype__),
        )

        if rtk_connection is not None:
            second_fix = fix_type(
                header=_header(typestore, 1_200_000_000, 1),
                status=status_type(status=0, service=1),
                latitude=22.3001,
                longitude=113.9001,
                altitude=80.5,
                position_covariance=np.zeros(9, dtype=np.float64),
                position_covariance_type=0,
            )
            writer.write(
                rtk_connection,
                1_200_000_000,
                typestore.serialize_ros1(second_fix, second_fix.__msgtype__),
            )

        second_image = compressed_type(
            header=_header(typestore, 0, 1),
            format="jpeg",
            data=np.frombuffer(jpeg_payload, dtype=np.uint8).copy(),
        )
        writer.write(
            image_connection,
            1_300_000_000,
            typestore.serialize_ros1(second_image, second_image.__msgtype__),
        )

    return jpeg_payload


def test_export_preserves_image_bytes_and_writes_compatible_csv(
    tmp_path: Path,
) -> None:
    """Экспорт не перекодирует JPEG и явно учитывает fallback нулевого stamp."""

    bag_path = tmp_path / "synthetic.bag"
    jpeg_payload = _write_synthetic_bag(bag_path)
    output_directory = tmp_path / "exported"

    summary = export_mars_lvig_bag(
        bag_path,
        output_directory,
        camera_topic=CAMERA_TOPIC,
        rtk_topic=RTK_TOPIC,
        route_start_offset_seconds=0.0,
    )

    assert summary.image_count == 2
    assert summary.rtk_count == 2
    assert summary.image_header_timestamp_count == 1
    assert summary.image_bag_timestamp_fallback_count == 1
    assert (output_directory / "images/000000.jpg").read_bytes() == jpeg_payload
    assert (output_directory / "images/000001.jpg").read_bytes() == jpeg_payload

    with (output_directory / "image_ids.csv").open(newline="") as csv_file:
        image_rows = list(csv.reader(csv_file))
    with (output_directory / "gps.csv").open(newline="") as csv_file:
        gps_rows = list(csv.reader(csv_file))
    assert image_rows == [
        ["Timestamp", "Image Id"],
        ["1100000000", "0"],
        ["1300000000", "1"],
    ]
    assert gps_rows[0] == ["Timestamp", "latitude", "longitude", "altitude"]
    assert len(gps_rows) == 3


def test_export_refuses_bag_without_rtk_topic(tmp_path: Path) -> None:
    """Отсутствие контрольной истины должно останавливать, а не украшать опыт."""

    bag_path = tmp_path / "camera_only.bag"
    _write_synthetic_bag(bag_path, include_rtk=False)

    with pytest.raises(RuntimeError, match="нет темы RTK"):
        export_mars_lvig_bag(
            bag_path,
            tmp_path / "exported",
            camera_topic=CAMERA_TOPIC,
            rtk_topic=RTK_TOPIC,
            route_start_offset_seconds=0.0,
        )


def _write_synthetic_mcap(directory: Path) -> tuple[Path, bytes]:
    """Пишет тот же минимальный поток в ROS2 MCAP, как у выбранного зеркала."""

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    time_type = typestore.types["builtin_interfaces/msg/Time"]
    header_type = typestore.types["std_msgs/msg/Header"]
    compressed_type = typestore.types["sensor_msgs/msg/CompressedImage"]
    fix_type = typestore.types["sensor_msgs/msg/NavSatFix"]
    status_type = typestore.types["sensor_msgs/msg/NavSatStatus"]
    jpeg_payload = b"\xff\xd8\xffsynthetic-mcap-jpeg\xff\xd9"

    def header(timestamp_nanoseconds: int) -> object:
        seconds, nanoseconds = divmod(timestamp_nanoseconds, 1_000_000_000)
        return header_type(
            stamp=time_type(sec=seconds, nanosec=nanoseconds),
            frame_id="test",
        )

    with Writer2(
        directory,
        version=9,
        storage_plugin=StoragePlugin.MCAP,
    ) as writer:
        image_connection = writer.add_connection(
            CAMERA_TOPIC,
            "sensor_msgs/msg/CompressedImage",
            typestore=typestore,
        )
        rtk_connection = writer.add_connection(
            RTK_TOPIC,
            "sensor_msgs/msg/NavSatFix",
            typestore=typestore,
        )
        messages = [
            (
                rtk_connection,
                1_000_000_000,
                fix_type(
                    header=header(1_000_000_000),
                    status=status_type(status=0, service=1),
                    latitude=22.30,
                    longitude=113.90,
                    altitude=80.0,
                    position_covariance=np.zeros(9, dtype=np.float64),
                    position_covariance_type=0,
                ),
            ),
            (
                image_connection,
                1_100_000_000,
                compressed_type(
                    header=header(1_100_000_000),
                    format="jpeg",
                    data=np.frombuffer(jpeg_payload, dtype=np.uint8).copy(),
                ),
            ),
            (
                rtk_connection,
                1_200_000_000,
                fix_type(
                    header=header(1_200_000_000),
                    status=status_type(status=0, service=1),
                    latitude=22.3001,
                    longitude=113.9001,
                    altitude=80.5,
                    position_covariance=np.zeros(9, dtype=np.float64),
                    position_covariance_type=0,
                ),
            ),
        ]
        for connection, timestamp, message in messages:
            writer.write(
                connection,
                timestamp,
                typestore.serialize_cdr(message, message.__msgtype__),
            )

    mcap_files = list(directory.glob("*.mcap"))
    assert len(mcap_files) == 1
    return mcap_files[0], jpeg_payload


def test_export_reads_standalone_ros2_mcap(tmp_path: Path) -> None:
    """MCAP-зеркало должно читаться напрямую, без установленного ROS2."""

    mcap_path, jpeg_payload = _write_synthetic_mcap(tmp_path / "mcap_bag")
    output_directory = tmp_path / "mcap_exported"

    summary = export_mars_lvig_bag(
        mcap_path,
        output_directory,
        camera_topic=CAMERA_TOPIC,
        rtk_topic=RTK_TOPIC,
        route_start_offset_seconds=0.0,
    )

    assert summary.image_count == 1
    assert summary.rtk_count == 2
    assert (output_directory / "images/000000.jpg").read_bytes() == jpeg_payload
