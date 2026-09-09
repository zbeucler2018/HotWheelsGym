"""Dino Boneyard reference line derived from native racer telemetry.

The table stores one median X/Z point per modulo-342 progress unit. It was
sampled from all three stock CPU racers over repeated laps on the supported ROM,
then the single unobserved unit (337) was linearly interpolated. No ROM bytes or
framebuffer imagery are included.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, pi, sin

from .npc_control import HEADING_PERIOD, RacerState

DINO_TRACK_POINT_COUNT = 342
DINO_TRACK_LATERAL_SCALE = 1 << 20
DINO_TRACK_LOOKAHEADS = (4, 12, 24)

DINO_TRACK_OBSERVATION_NAMES = (
    "track_lateral_offset",
    "track_heading_error_sin",
    "track_heading_error_cos",
    "track_target_short_forward",
    "track_target_short_right",
    "track_target_medium_forward",
    "track_target_medium_right",
    "track_target_long_forward",
    "track_target_long_right",
    "track_curvature_medium",
    "track_curvature_long",
)
DINO_TRACK_OBSERVATION_SIZE = len(DINO_TRACK_OBSERVATION_NAMES)

# Coordinates use the signed 24.8-ish world units exposed by racer +0xF8/+0x100.
DINO_BONEYARD_CENTERLINE = (
    (7108960, 16260240),  # 0
    (7416045, 16260240),  # 1
    (7696735, 16259933),  # 2
    (7925796, 16227815),  # 3
    (8213385, 16215105),  # 4
    (8453014, 16158005),  # 5
    (8659303, 16080178),  # 6
    (8858784, 15962880),  # 7
    (9056891, 15825934),  # 8
    (9284217, 15644451),  # 9
    (9516429, 15445026),  # 10
    (9684125, 15288666),  # 11
    (9799129, 15111358),  # 12
    (9918865, 14932020),  # 13
    (9900296, 14700269),  # 14
    (9973814, 14470342),  # 15
    (10007930, 14178248),  # 16
    (10039585, 13853460),  # 17
    (10083696, 13528564),  # 18
    (10117564, 13201312),  # 19
    (10146478, 12866665),  # 20
    (10172294, 12548920),  # 21
    (10197563, 12229364),  # 22
    (10223737, 11993406),  # 23
    (10226780, 11816707),  # 24
    (10218136, 11547963),  # 25
    (10194632, 11312370),  # 26
    (10225785, 11213951),  # 27
    (10228292, 11067528),  # 28
    (10228603, 10910317),  # 29
    (10229678, 10755971),  # 30
    (10221614, 10707566),  # 31
    (10193927, 10600314),  # 32
    (9904434, 10656386),  # 33
    (9870986, 10549604),  # 34
    (9822407, 10455382),  # 35
    (9776598, 9655409),  # 36
    (9785737, 9335496),  # 37
    (9800393, 9008188),  # 38
    (9821675, 8675141),  # 39
    (9843624, 8348845),  # 40
    (9846612, 8027272),  # 41
    (9851256, 7807770),  # 42
    (9840332, 7570964),  # 43
    (9783691, 7357834),  # 44
    (9705094, 7267373),  # 45
    (9600502, 7201940),  # 46
    (9494523, 7169630),  # 47
    (9380772, 7168533),  # 48
    (9263254, 7186631),  # 49
    (9177174, 7227994),  # 50
    (9079448, 7318055),  # 51
    (9033512, 7416425),  # 52
    (8995036, 7641508),  # 53
    (8978266, 7854426),  # 54
    (8966226, 8111854),  # 55
    (8951059, 8414580),  # 56
    (8925662, 8687320),  # 57
    (8877988, 8944700),  # 58
    (8784241, 9175039),  # 59
    (8628828, 9401065),  # 60
    (8451832, 9611105),  # 61
    (8277561, 9816357),  # 62
    (8132440, 9993713),  # 63
    (8006796, 10161881),  # 64
    (7895420, 10374210),  # 65
    (7873558, 10556292),  # 66
    (7860920, 10797303),  # 67
    (7852867, 10994405),  # 68
    (7837736, 11288326),  # 69
    (7811318, 11552975),  # 70
    (7763390, 11805902),  # 71
    (7657555, 12053732),  # 72
    (7512369, 12271547),  # 73
    (7352918, 12473564),  # 74
    (7181058, 12664843),  # 75
    (6963168, 12861501),  # 76
    (6743027, 13003306),  # 77
    (6519597, 13083288),  # 78
    (6267748, 13082861),  # 79
    (6073403, 13048159),  # 80
    (5811916, 12921944),  # 81
    (5593909, 12777232),  # 82
    (5357620, 12600821),  # 83
    (5115558, 12401190),  # 84
    (4871927, 12182389),  # 85
    (4633927, 11957477),  # 86
    (4411864, 11735555),  # 87
    (4229696, 11529216),  # 88
    (4085713, 11327121),  # 89
    (3973302, 11076606),  # 90
    (3922454, 10853372),  # 91
    (3891669, 10609561),  # 92
    (3869525, 10332304),  # 93
    (3853984, 10024857),  # 94
    (3846900, 9710614),  # 95
    (3847409, 9369658),  # 96
    (3862583, 9105390),  # 97
    (3899242, 8849314),  # 98
    (4003010, 8574460),  # 99
    (4123776, 8379714),  # 100
    (4281299, 8170493),  # 101
    (4450704, 7967136),  # 102
    (4645966, 7746392),  # 103
    (4870121, 7503960),  # 104
    (5093154, 7267929),  # 105
    (5320772, 7032489),  # 106
    (5549646, 6798285),  # 107
    (5777768, 6563345),  # 108
    (5931987, 6398002),  # 109
    (6073670, 6225858),  # 110
    (6189010, 6036910),  # 111
    (6264384, 5873150),  # 112
    (6345639, 5727501),  # 113
    (6454340, 5571795),  # 114
    (6586196, 5422306),  # 115
    (6763434, 5282249),  # 116
    (6935060, 5187486),  # 117
    (7121018, 5110225),  # 118
    (7304821, 5055426),  # 119
    (7461953, 5010690),  # 120
    (7618310, 4954072),  # 121
    (7736029, 4892018),  # 122
    (7836950, 4812634),  # 123
    (7920580, 4715172),  # 124
    (7990187, 4598722),  # 125
    (8069151, 4460587),  # 126
    (8165194, 4305806),  # 127
    (8270480, 4162624),  # 128
    (8439044, 3962128),  # 129
    (8618016, 3764373),  # 130
    (8805732, 3562505),  # 131
    (9030890, 3324609),  # 132
    (9258700, 3089176),  # 133
    (9488718, 2856088),  # 134
    (9720912, 2625144),  # 135
    (9954566, 2395684),  # 136
    (10154905, 2205342),  # 137
    (10352129, 2033148),  # 138
    (10571967, 1891616),  # 139
    (10785257, 1809203),  # 140
    (11009144, 1760853),  # 141
    (11238248, 1729634),  # 142
    (11529788, 1706856),  # 143
    (11824228, 1693551),  # 144
    (12088314, 1686547),  # 145
    (12399164, 1686124),  # 146
    (12674558, 1702726),  # 147
    (12898758, 1748412),  # 148
    (13113182, 1833802),  # 149
    (13308358, 1949382),  # 150
    (13516344, 2108204),  # 151
    (13716420, 2277584),  # 152
    (13925811, 2464134),  # 153
    (14166328, 2686336),  # 154
    (14403594, 2912034),  # 155
    (14638724, 3139972),  # 156
    (14872782, 3368990),  # 157
    (15105888, 3599016),  # 158
    (15338898, 3829146),  # 159
    (15571004, 4060172),  # 160
    (15801958, 4292350),  # 161
    (16032212, 4525188),  # 162
    (16260766, 4759738),  # 163
    (16471509, 4985607),  # 164
    (16652033, 5200010),  # 165
    (16798056, 5417049),  # 166
    (16899686, 5657742),  # 167
    (16955260, 5913494),  # 168
    (16987123, 6187826),  # 169
    (16990824, 6420697),  # 170
    (16988698, 6696362),  # 171
    (16933191, 6966539),  # 172
    (16825156, 7187510),  # 173
    (16662172, 7367010),  # 174
    (16493489, 7471665),  # 175
    (16218072, 7567109),  # 176
    (15961360, 7619034),  # 177
    (15669718, 7660804),  # 178
    (15358965, 7690574),  # 179
    (15032133, 7710971),  # 180
    (14704893, 7724165),  # 181
    (14377493, 7732663),  # 182
    (14100223, 7737863),  # 183
    (13805509, 7739824),  # 184
    (13478021, 7739558),  # 185
    (13198645, 7738526),  # 186
    (12903931, 7738574),  # 187
    (12675125, 7749482),  # 188
    (12452129, 7780990),  # 189
    (12249529, 7856037),  # 190
    (12151362, 7940356),  # 191
    (12085706, 8054122),  # 192
    (12051794, 8263382),  # 193
    (12033191, 8508388),  # 194
    (12022460, 8707073),  # 195
    (12009675, 9061878),  # 196
    (12001499, 9394358),  # 197
    (11995515, 9684130),  # 198
    (11991468, 9964844),  # 199
    (11989091, 10275994),  # 200
    (11987212, 10535355),  # 201
    (11986139, 10731825),  # 202
    (11985651, 11026602),  # 203
    (11984708, 11600088),  # 204
    (11984859, 11659687),  # 205
    (11984489, 11667823),  # 206
    (11827071, 12093143),  # 207
    (11814060, 12264919),  # 208
    (11832427, 12360315),  # 209
    (11896302, 12719897),  # 210
    (11924867, 13042078),  # 211
    (11941768, 13339216),  # 212
    (11953406, 13665774),  # 213
    (11966201, 13896578),  # 214
    (11986717, 14112740),  # 215
    (12042877, 14324002),  # 216
    (12160687, 14500198),  # 217
    (12331898, 14645726),  # 218
    (12488738, 14806857),  # 219
    (12666292, 15006881),  # 220
    (12803530, 15227926),  # 221
    (12937060, 15456558),  # 222
    (13034990, 15621594),  # 223
    (13131012, 15779796),  # 224
    (13238890, 15902598),  # 225
    (13333016, 15998072),  # 226
    (13436546, 16072682),  # 227
    (13552185, 16126103),  # 228
    (13665544, 16159074),  # 229
    (13837122, 16200857),  # 230
    (13997440, 16248252),  # 231
    (14164484, 16315374),  # 232
    (14337266, 16408172),  # 233
    (14510546, 16536608),  # 234
    (14642840, 16681163),  # 235
    (14767664, 16873247),  # 236
    (14855188, 17027210),  # 237
    (14936288, 17169429),  # 238
    (15036770, 17322473),  # 239
    (15164958, 17488471),  # 240
    (15296313, 17659873),  # 241
    (15386766, 17796227),  # 242
    (15524083, 18056272),  # 243
    (15623896, 18381756),  # 244
    (15679021, 18657422),  # 245
    (15723403, 18962350),  # 246
    (15748184, 19340395),  # 247
    (15750070, 19615730),  # 248
    (15729924, 19909124),  # 249
    (15655174, 20193616),  # 250
    (15556232, 20420706),  # 251
    (15431833, 20612734),  # 252
    (15258786, 20808688),  # 253
    (15080420, 20951931),  # 254
    (14856234, 21078620),  # 255
    (14638391, 21159355),  # 256
    (14382436, 21214238),  # 257
    (14107939, 21248439),  # 258
    (13828630, 21267231),  # 259
    (13534299, 21284213),  # 260
    (13309255, 21302714),  # 261
    (13114721, 21329835),  # 262
    (12927195, 21392848),  # 263
    (12772354, 21506406),  # 264
    (12638824, 21630822),  # 265
    (12504919, 21754840),  # 266
    (12265387, 21974634),  # 267
    (12050883, 22173258),  # 268
    (11814301, 22383910),  # 269
    (11593702, 22544286),  # 270
    (11364100, 22669610),  # 271
    (11170398, 22723557),  # 272
    (10911886, 22766012),  # 273
    (10635055, 22796416),  # 274
    (10308363, 22819439),  # 275
    (9970240, 22833390),  # 276
    (9656554, 22839971),  # 277
    (9149657, 22839578),  # 278
    (9042516, 22837788),  # 279
    (8771704, 22809509),  # 280
    (8502705, 22721544),  # 281
    (8332312, 22631256),  # 282
    (8121666, 22448805),  # 283
    (7911106, 22302146),  # 284
    (7676587, 22094570),  # 285
    (7505923, 21890410),  # 286
    (7385049, 21670636),  # 287
    (7333715, 21467700),  # 288
    (7275211, 21177378),  # 289
    (7219641, 20959379),  # 290
    (7109933, 20649086),  # 291
    (6987904, 20455143),  # 292
    (6813925, 20233526),  # 293
    (6619062, 20014097),  # 294
    (6403374, 19782804),  # 295
    (6231958, 19630665),  # 296
    (6059868, 19511536),  # 297
    (5952714, 19485597),  # 298
    (5728738, 19467236),  # 299
    (5508822, 19456226),  # 300
    (5178484, 19445799),  # 301
    (4844079, 19438726),  # 302
    (4516619, 19434274),  # 303
    (4182206, 19431359),  # 304
    (3829918, 19428128),  # 305
    (3475788, 19417175),  # 306
    (3151766, 19387507),  # 307
    (2862480, 19331830),  # 308
    (2618194, 19243594),  # 309
    (2409332, 19131732),  # 310
    (2207098, 18976180),  # 311
    (2044552, 18805588),  # 312
    (1906204, 18590570),  # 313
    (1813141, 18378694),  # 314
    (1745636, 18118622),  # 315
    (1710786, 17850557),  # 316
    (1710350, 17573471),  # 317
    (1741516, 17312763),  # 318
    (1820679, 17064404),  # 319
    (1924986, 16858105),  # 320
    (2082077, 16650946),  # 321
    (2249252, 16493110),  # 322
    (2454292, 16360542),  # 323
    (2679434, 16261528),  # 324
    (2919881, 16200059),  # 325
    (3192148, 16156125),  # 326
    (3504171, 16123149),  # 327
    (3779107, 16101172),  # 328
    (4073020, 16086405),  # 329
    (4303044, 16078911),  # 330
    (4564075, 16073053),  # 331
    (4761418, 16070213),  # 332
    (5055250, 16068759),  # 333
    (5531282, 16066533),  # 334
    (5571334, 16070465),  # 335
    (5602858, 16191238),  # 336
    (5963995, 16225952),  # 337
    (6325132, 16260667),  # 338
    (6413661, 16260240),  # 339
    (6794361, 16260240),  # 340
    (6991585, 16260240),  # 341
)

if len(DINO_BONEYARD_CENTERLINE) != DINO_TRACK_POINT_COUNT:
    raise RuntimeError("Dino Boneyard centerline length does not match progress count")


@dataclass(frozen=True)
class DinoTrackPose:
    """Track-relative geometry for one racer."""

    progress_index: int
    center_x: float
    center_z: float
    tangent_x: float
    tangent_z: float
    lateral_offset: float
    heading_error_sin: float
    heading_error_cos: float
    features: tuple[float, ...]


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _unit(dx: float, dz: float) -> tuple[float, float]:
    length = hypot(dx, dz)
    if length <= 1e-9:
        return 0.0, 1.0
    return dx / length, dz / length


def _segment_projection(
    x: float, z: float, start: tuple[int, int], end: tuple[int, int]
) -> tuple[float, float, float, float, float]:
    dx = float(end[0] - start[0])
    dz = float(end[1] - start[1])
    length_squared = dx * dx + dz * dz
    if length_squared <= 1e-9:
        projected_x = float(start[0])
        projected_z = float(start[1])
        tangent_x, tangent_z = 0.0, 1.0
    else:
        fraction = max(
            0.0,
            min(
                1.0,
                ((x - start[0]) * dx + (z - start[1]) * dz) / length_squared,
            ),
        )
        projected_x = start[0] + fraction * dx
        projected_z = start[1] + fraction * dz
        tangent_x, tangent_z = _unit(dx, dz)
    distance_squared = (x - projected_x) ** 2 + (z - projected_z) ** 2
    return distance_squared, projected_x, projected_z, tangent_x, tangent_z


def _track_tangent(index: int, radius: int = 2) -> tuple[float, float]:
    before = DINO_BONEYARD_CENTERLINE[(index - radius) % DINO_TRACK_POINT_COUNT]
    after = DINO_BONEYARD_CENTERLINE[(index + radius) % DINO_TRACK_POINT_COUNT]
    return _unit(after[0] - before[0], after[1] - before[1])


def _target_direction(
    x: float,
    z: float,
    forward_x: float,
    forward_z: float,
    right_x: float,
    right_z: float,
    index: int,
) -> tuple[float, float]:
    target = DINO_BONEYARD_CENTERLINE[index % DINO_TRACK_POINT_COUNT]
    direction_x, direction_z = _unit(target[0] - x, target[1] - z)
    return (
        _clip(direction_x * forward_x + direction_z * forward_z),
        _clip(direction_x * right_x + direction_z * right_z),
    )


def dino_track_pose(state: RacerState, search_radius: int = 4) -> DinoTrackPose:
    """Project a racer onto the local reference line and expose steering cues."""

    nominal = state.progress % DINO_TRACK_POINT_COUNT
    best: tuple[float, int, float, float, float, float] | None = None
    for offset in range(-search_radius, search_radius + 1):
        index = (nominal + offset) % DINO_TRACK_POINT_COUNT
        start = DINO_BONEYARD_CENTERLINE[index]
        end = DINO_BONEYARD_CENTERLINE[(index + 1) % DINO_TRACK_POINT_COUNT]
        distance, center_x, center_z, tangent_x, tangent_z = _segment_projection(
            state.x, state.z, start, end
        )
        candidate = (distance, index, center_x, center_z, tangent_x, tangent_z)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    _, index, center_x, center_z, tangent_x, tangent_z = best

    angle = 2.0 * pi * (state.current_heading & (HEADING_PERIOD - 1)) / HEADING_PERIOD
    forward_x, forward_z = sin(angle), cos(angle)
    right_x, right_z = cos(angle), -sin(angle)
    lateral = _clip(
        ((state.x - center_x) * tangent_z - (state.z - center_z) * tangent_x)
        / DINO_TRACK_LATERAL_SCALE
    )
    heading_error_sin = _clip(tangent_x * right_x + tangent_z * right_z)
    heading_error_cos = _clip(tangent_x * forward_x + tangent_z * forward_z)

    targets: list[float] = []
    for lookahead in DINO_TRACK_LOOKAHEADS:
        targets.extend(
            _target_direction(
                state.x,
                state.z,
                forward_x,
                forward_z,
                right_x,
                right_z,
                index + lookahead,
            )
        )

    medium_tangent = _track_tangent(index + DINO_TRACK_LOOKAHEADS[1])
    long_tangent = _track_tangent(index + DINO_TRACK_LOOKAHEADS[2])
    curvature_medium = _clip(
        tangent_x * medium_tangent[1] - tangent_z * medium_tangent[0]
    )
    curvature_long = _clip(tangent_x * long_tangent[1] - tangent_z * long_tangent[0])
    features = (
        lateral,
        heading_error_sin,
        heading_error_cos,
        *targets,
        curvature_medium,
        curvature_long,
    )
    if len(features) != DINO_TRACK_OBSERVATION_SIZE:
        raise RuntimeError("Dino track feature name and value counts differ")
    return DinoTrackPose(
        progress_index=index,
        center_x=center_x,
        center_z=center_z,
        tangent_x=tangent_x,
        tangent_z=tangent_z,
        lateral_offset=lateral,
        heading_error_sin=heading_error_sin,
        heading_error_cos=heading_error_cos,
        features=features,
    )
