import matplotlib.pyplot as plt
from descartes import PolygonPatch
from shapely.geometry import LineString


def main(tracks):
    track1 = LineString([[p[1], p[0]] for p in tracks[0]])
    track2 = LineString([[p[1], p[0]] for p in tracks[1]])
    track1_buffered = track1.buffer(5)
    # fig=plt.figure()
    # ax = fig.add_subplot(111)
    patch1 = PolygonPatch(
        track1_buffered, fc="blue", ec="blue", alpha=0.5, zorder=2
    )
    # ax.add_patch(patch1)
    # x,y=track1.xy
    # ax.plot(x,y,'b.')
    # x,y=track2.xy
    # ax.plot(x,y,'g.')
    # plt.show()
    match = track1_buffered.intersection(track2).buffer(5)
    print("Match: ", dir(match))
    # print("Match: ", match.simplify(0.5))
    # print("Match: ", match.geoms.shape_factory())
    match = match.simplify(0.75, preserve_topology=False)
    print("Match: ", match)
    # print("Envelope: ", match.envelope)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch1 = PolygonPatch(match, fc="green", ec="green", alpha=0.5, zorder=2)
    ax.add_patch(patch1)
    x, y = track1.xy
    ax.plot(x, y, "b.")
    x, y = track2.xy
    ax.plot(x, y, "g.")
    plt.show()


if __name__ == "__main__":
    tracks = [
        [
            (119, 10),
            (118, 22),
            (118, 35),
            (119, 47),
            (121, 60),
            (124, 72),
            (128, 84),
            (133, 95),
            (139, 106),
            (145, 117),
            (152, 127),
            (159, 137),
            (167, 146),
            (176, 156),
            (184, 165),
            (193, 175),
            (202, 183),
            (210, 193),
            (219, 201),
            (228, 211),
            (236, 220),
            (244, 230),
            (252, 239),
            (259, 249),
            (266, 259),
            (272, 270),
            (278, 281),
            (283, 293),
            (286, 305),
            (289, 317),
            (290, 330),
            (289, 342),
            (287, 354),
            (283, 366),
            (277, 377),
            (269, 387),
            (259, 395),
            (248, 401),
            (236, 404),
            (224, 404),
            (212, 403),
            (200, 399),
            (189, 392),
            (179, 385),
            (170, 376),
            (162, 367),
            (157, 355),
            (152, 343),
            (148, 331),
            (145, 319),
            (144, 307),
            (142, 295),
            (142, 282),
        ],
        [
            (299, 30),
            (290, 21),
            (280, 14),
            (269, 8),
            (257, 4),
            (244, 2),
            (232, 1),
            (220, 2),
            (208, 5),
            (196, 9),
            (185, 15),
            (175, 23),
            (167, 32),
            (159, 42),
            (153, 53),
            (149, 65),
            (147, 78),
            (146, 90),
            (147, 102),
            (150, 115),
            (155, 126),
            (162, 137),
            (169, 147),
            (176, 156),
            (185, 166),
            (194, 174),
            (202, 183),
            (212, 191),
            (220, 200),
            (229, 209),
            (237, 219),
            (244, 231),
            (248, 242),
            (252, 253),
            (253, 266),
            (253, 279),
            (250, 291),
            (246, 303),
            (241, 314),
            (234, 324),
            (225, 333),
            (215, 340),
            (204, 347),
            (193, 351),
            (180, 354),
            (168, 355),
            (156, 353),
            (143, 351),
            (132, 346),
            (121, 340),
        ],
    ]
    main(tracks)
