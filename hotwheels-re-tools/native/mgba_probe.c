/*
 * Headless mGBA probe for the historical HotWheelsGym savestates.
 *
 * This deliberately links against the public mGBA core API. It does not
 * contain or locate a ROM or BIOS. Generated images and patched ROMs should
 * remain outside version control.
 */

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mgba/core/core.h>

enum {
    WIDTH = 240,
    HEIGHT = 160,
    STATE_SIZE = 0x61000,
    DEFAULT_MANAGER = 0x020068F8,
    MANAGER_PLAYER_COUNT = 0x448,
    MANAGER_CPU_COUNT = 0x449,
    MANAGER_TOTAL_COUNT = 0x44A,
    MANAGER_POINTER_LIST = 0x450,
    RACER_CURRENT_HEADING = 0xDE,
    RACER_DESIRED_HEADING = 0xE0,
    RACER_X = 0xF8,
    RACER_Z = 0x100,
    RACER_PROGRESS = 0x148,
    RACER_SPEED = 0xE8,
    RACER_TARGET_SPEED = 0x2F0,
};

struct Options {
    uint32_t manager;
    int resume_old_hle;
    int heading_enabled;
    unsigned heading_slot;
    uint16_t heading;
    int speed_enabled;
    unsigned speed_slot;
    uint32_t speed;
};

static uint32_t read_le32(const uint8_t* data) {
    return (uint32_t) data[0] | (uint32_t) data[1] << 8 |
           (uint32_t) data[2] << 16 | (uint32_t) data[3] << 24;
}

static int parse_u32(const char* text, uint32_t* output) {
    char* end = NULL;
    unsigned long value = strtoul(text, &end, 0);
    if (!text[0] || !end || *end || value > UINT32_MAX) {
        return 0;
    }
    *output = (uint32_t) value;
    return 1;
}

static int parse_heading(const char* text, unsigned* slot, uint16_t* heading) {
    char* separator = strchr(text, ':');
    char* end = NULL;
    if (!separator) {
        return 0;
    }
    unsigned long parsed_slot = strtoul(text, &end, 0);
    if (end != separator || parsed_slot > UINT32_MAX) {
        return 0;
    }
    unsigned long parsed_heading = strtoul(separator + 1, &end, 0);
    if (!separator[1] || !end || *end || parsed_heading > 0xFFF) {
        return 0;
    }
    *slot = (unsigned) parsed_slot;
    *heading = (uint16_t) parsed_heading;
    return 1;
}

static int parse_slot_u32(const char* text, unsigned* slot, uint32_t* value) {
    char* separator = strchr(text, ':');
    char* end = NULL;
    if (!separator) {
        return 0;
    }
    unsigned long parsed_slot = strtoul(text, &end, 0);
    if (end != separator || parsed_slot > UINT32_MAX) {
        return 0;
    }
    unsigned long parsed_value = strtoul(separator + 1, &end, 0);
    if (!separator[1] || !end || *end || parsed_value > UINT32_MAX) {
        return 0;
    }
    *slot = (unsigned) parsed_slot;
    *value = (uint32_t) parsed_value;
    return 1;
}

static int load_state(const char* path, uint8_t* output, size_t size) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "cannot open state: %s\n", path);
        return 0;
    }
    size_t total = 0;
    while (total < size) {
        size_t count = fread(output + total, 1, size - total, file);
        if (!count) {
            break;
        }
        total += count;
    }
    uint8_t extra = 0;
    size_t has_extra = fread(&extra, 1, 1, file);
    int read_error = ferror(file);
    fclose(file);
    if (read_error) {
        fprintf(stderr, "cannot read state: %s\n", path);
        return 0;
    }
    if (total != size || has_extra != 0) {
        fprintf(stderr, "state expanded to %zu bytes, expected 0x%X\n", total, STATE_SIZE);
        return 0;
    }
    return 1;
}

static int write_ppm(const char* path, const color_t* pixels) {
    FILE* file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "cannot write %s: %s\n", path, strerror(errno));
        return 0;
    }
    fprintf(file, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            color_t pixel = pixels[y * WIDTH + x];
            uint8_t rgb[3] = {
                (uint8_t) pixel,
                (uint8_t) (pixel >> 8),
                (uint8_t) (pixel >> 16),
            };
            if (fwrite(rgb, sizeof(rgb), 1, file) != 1) {
                fprintf(stderr, "cannot finish %s\n", path);
                fclose(file);
                return 0;
            }
        }
    }
    return fclose(file) == 0;
}

static uint32_t racer_address(struct mCore* core, uint32_t manager, unsigned slot) {
    uint32_t pointer_list = core->busRead32(core, manager + MANAGER_POINTER_LIST);
    return core->busRead32(core, pointer_list + slot * 4);
}

static void print_racers(struct mCore* core, uint32_t manager, const char* label) {
    unsigned players = core->busRead8(core, manager + MANAGER_PLAYER_COUNT);
    unsigned cpus = core->busRead8(core, manager + MANAGER_CPU_COUNT);
    unsigned total = core->busRead8(core, manager + MANAGER_TOTAL_COUNT);
    uint32_t pointer_list = core->busRead32(core, manager + MANAGER_POINTER_LIST);
    printf("%s frame=%" PRIu32 " manager=%08" PRIx32
           " list=%08" PRIx32 " players=%u cpus=%u total=%u\n",
           label, core->frameCounter(core), manager, pointer_list, players, cpus, total);
    if (!total || total > 16 || pointer_list < 0x02000000 || pointer_list >= 0x02040000) {
        printf("  racer table is not plausible\n");
        return;
    }
    for (unsigned slot = 0; slot < total; ++slot) {
        uint32_t racer = racer_address(core, manager, slot);
        printf("  slot=%u address=%08" PRIx32 " progress=%u heading=%03x"
               " desired=%03x speed=%" PRId32 " target_speed=%" PRId32
               " x=%" PRId32 " z=%" PRId32 "\n",
               slot, racer, core->busRead16(core, racer + RACER_PROGRESS),
               core->busRead16(core, racer + RACER_CURRENT_HEADING),
               core->busRead16(core, racer + RACER_DESIRED_HEADING),
               (int32_t) core->busRead32(core, racer + RACER_SPEED),
               (int32_t) core->busRead32(core, racer + RACER_TARGET_SPEED),
               (int32_t) core->busRead32(core, racer + RACER_X),
               (int32_t) core->busRead32(core, racer + RACER_Z));
    }
}

static int resume_old_hle_irq(struct mCore* core, const uint8_t* state) {
    uint32_t pc = 0;
    uint32_t lr = 0;
    uint32_t cpsr = 0;
    uint32_t spsr = read_le32(state + 0x64);
    core->readRegister(core, "pc", &pc);
    core->readRegister(core, "lr", &lr);
    core->readRegister(core, "cpsr", &cpsr);
    if (pc >= 0x4000 || (cpsr & 0x1F) != 0x12 || (spsr & 0x1F) == 0x12 || lr < 4) {
        fprintf(stderr,
                "refusing IRQ recovery for pc=%08" PRIx32 " lr=%08" PRIx32
                " cpsr=%08" PRIx32 " spsr=%08" PRIx32 "\n",
                pc, lr, cpsr, spsr);
        return 0;
    }
    pc = lr - 4;
    if (!core->writeRegister(core, "cpsr", &spsr) ||
        !core->writeRegister(core, "pc", &pc)) {
        fprintf(stderr, "failed to rewrite PC/CPSR\n");
        return 0;
    }
    printf("old-HLE IRQ recovery: pc=%08" PRIx32 " cpsr=%08" PRIx32 "\n", pc, spsr);
    return 1;
}

static void usage(const char* executable) {
    fprintf(stderr,
            "usage: %s ROM STATE FRAMES OUTPUT.ppm [--resume-old-hle]"
            " [--manager ADDRESS] [--cpu-heading SLOT:ANGLE]"
            " [--cpu-target-speed SLOT:VALUE]\n",
            executable);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        usage(argv[0]);
        return 2;
    }
    char* frame_end = NULL;
    long frames = strtol(argv[3], &frame_end, 0);
    if (!argv[3][0] || !frame_end || *frame_end || frames < 0) {
        fprintf(stderr, "invalid frame count: %s\n", argv[3]);
        return 2;
    }
    struct Options options = {DEFAULT_MANAGER, 0, 0, 0, 0, 0, 0, 0};
    for (int index = 5; index < argc; ++index) {
        if (!strcmp(argv[index], "--resume-old-hle")) {
            options.resume_old_hle = 1;
        } else if (!strcmp(argv[index], "--manager") && index + 1 < argc) {
            if (!parse_u32(argv[++index], &options.manager)) {
                fprintf(stderr, "invalid manager address: %s\n", argv[index]);
                return 2;
            }
        } else if (!strcmp(argv[index], "--cpu-heading") && index + 1 < argc) {
            if (!parse_heading(argv[++index], &options.heading_slot, &options.heading)) {
                fprintf(stderr, "invalid SLOT:ANGLE: %s\n", argv[index]);
                return 2;
            }
            options.heading_enabled = 1;
        } else if (!strcmp(argv[index], "--cpu-target-speed") && index + 1 < argc) {
            if (!parse_slot_u32(argv[++index], &options.speed_slot, &options.speed)) {
                fprintf(stderr, "invalid SLOT:VALUE: %s\n", argv[index]);
                return 2;
            }
            options.speed_enabled = 1;
        } else {
            fprintf(stderr, "unknown or incomplete option: %s\n", argv[index]);
            usage(argv[0]);
            return 2;
        }
    }

    uint8_t* state = malloc(STATE_SIZE);
    color_t* pixels = calloc(WIDTH * HEIGHT, sizeof(*pixels));
    if (!state || !pixels) {
        fprintf(stderr, "allocation failed\n");
        free(state);
        free(pixels);
        return 1;
    }
    if (!load_state(argv[2], state, STATE_SIZE)) {
        free(state);
        free(pixels);
        return 1;
    }
    printf("state version=%08" PRIx32 " bios=%08" PRIx32 " rom_crc32=%08" PRIx32 "\n",
           read_le32(state), read_le32(state + 4), read_le32(state + 8));

    struct mCore* core = mCoreFind(argv[1]);
    if (!core || !core->init(core)) {
        fprintf(stderr, "mGBA could not initialize a core for %s\n", argv[1]);
        free(state);
        free(pixels);
        return 1;
    }
    mCoreInitConfig(core, NULL);
    core->setVideoBuffer(core, pixels, WIDTH);
    if (!mCoreLoadFile(core, argv[1])) {
        fprintf(stderr, "ROM load failed\n");
        core->deinit(core);
        free(state);
        free(pixels);
        return 1;
    }
    core->reset(core);
    if (!core->loadState(core, state)) {
        fprintf(stderr, "savestate load failed\n");
        core->unloadROM(core);
        core->deinit(core);
        free(state);
        free(pixels);
        return 1;
    }
    if (options.resume_old_hle && !resume_old_hle_irq(core, state)) {
        core->unloadROM(core);
        core->deinit(core);
        free(state);
        free(pixels);
        return 1;
    }

    unsigned total = core->busRead8(core, options.manager + MANAGER_TOTAL_COUNT);
    if (options.heading_enabled && options.heading_slot >= total) {
        fprintf(stderr, "CPU heading slot %u is outside the %u-racer table\n",
                options.heading_slot, total);
        core->unloadROM(core);
        core->deinit(core);
        free(state);
        free(pixels);
        return 1;
    }
    if (options.speed_enabled && options.speed_slot >= total) {
        fprintf(stderr, "CPU speed slot %u is outside the %u-racer table\n",
                options.speed_slot, total);
        core->unloadROM(core);
        core->deinit(core);
        free(state);
        free(pixels);
        return 1;
    }
    uint32_t controlled_racer = options.heading_enabled
                                    ? racer_address(core, options.manager, options.heading_slot)
                                    : 0;
    uint32_t speed_racer = options.speed_enabled
                               ? racer_address(core, options.manager, options.speed_slot)
                               : 0;
    print_racers(core, options.manager, "before");
    for (long frame = 0; frame < frames; ++frame) {
        if (options.heading_enabled) {
            core->busWrite16(core, controlled_racer + RACER_DESIRED_HEADING,
                             options.heading);
        }
        if (options.speed_enabled) {
            core->busWrite32(core, speed_racer + RACER_TARGET_SPEED, options.speed);
        }
        core->setKeys(core, 0);
        core->runFrame(core);
    }
    print_racers(core, options.manager, "after");

    int wrote = write_ppm(argv[4], pixels);
    core->unloadROM(core);
    core->deinit(core);
    free(state);
    free(pixels);
    return wrote ? 0 : 1;
}
