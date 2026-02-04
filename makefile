# Tools
CC      = clang
LEX     = lex
YACC    = yacc

# Flags
INC_DIR = include
WFLAGS  = -Wall -Wextra -Wpedantic -Wno-unknown-pragmas
OFLAGS  = -O3 -march=native -ffast-math -fno-finite-math-only
CFLAGS  = $(OFLAGS) $(WFLAGS) -I$(INC_DIR)
LDFLAGS = -lm
EXEC    = strasgpt

# Layout
SRC_DIR   = source
BUILD_DIR = build

# Sources and objects
SRC_C   = $(wildcard $(SRC_DIR)/*.c)
OBJ_C   = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRC_C))
GEN_C   = $(BUILD_DIR)/y.tab.c $(BUILD_DIR)/lex.json_scanner_.c
GEN_O   = $(BUILD_DIR)/y.tab.o $(BUILD_DIR)/lex.json_scanner_.o
OBJ     = $(OBJ_C) $(GEN_O)

.PHONY: all clean debug

all: $(EXEC)

# Final link
$(EXEC): $(OBJ) | $(BUILD_DIR)
	$(CC) -o $@ $(OBJ) $(LDFLAGS)

# Yacc: generate y.tab.c and y.tab.h into build/
$(BUILD_DIR)/y.tab.c $(BUILD_DIR)/y.tab.h: \
	$(SRC_DIR)/json_parser.y \
	$(INC_DIR)/safetensors.h \
	$(INC_DIR)/tokenizer.h
	mkdir -p $(BUILD_DIR)
	$(YACC) -d -o $(BUILD_DIR)/y.tab.c $(SRC_DIR)/json_parser.y

$(BUILD_DIR)/y.tab.o: $(BUILD_DIR)/y.tab.c
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

# Lex: generate scanner into build/ (note we disable -Wextra here)
$(BUILD_DIR)/lex.json_scanner_.c: \
	$(SRC_DIR)/json_scanner.l \
	$(BUILD_DIR)/y.tab.h
	mkdir -p $(BUILD_DIR)
	$(LEX) -o $@ $(SRC_DIR)/json_scanner.l

$(BUILD_DIR)/lex.json_scanner_.o: CFLAGS := $(filter-out -Wextra,$(CFLAGS))
$(BUILD_DIR)/lex.json_scanner_.o: \
	$(BUILD_DIR)/lex.json_scanner_.c
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

# C sources from source/ -> build/*.o
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

# Ensure build/ exists
$(BUILD_DIR):
	mkdir -p $@

# Auto-generated header dependencies
DEPS := $(OBJ:.o=.d)
-include $(DEPS)

clean:
	/bin/rm -rf $(BUILD_DIR) $(EXEC)

# Parallel MPI + OpenMP build
BREW_LIBOMP_DIR = /opt/homebrew/opt/libomp/lib
ifneq ($(wildcard $(BREW_LIBOMP_DIR)),)
BREW_LIBOMP_LD_FLAGS = -L$(BREW_LIBOMP_DIR) -lomp
else
BREW_LIBOMP_LD_FLAGS = -fopenmp
endif
parallel: CC = mpicc
parallel: CFLAGS = $(OFLAGS) -fopenmp $(WFLAGS) -I$(INC_DIR) -DPARALLEL=1
parallel: LDFLAGS = $(BREW_LIBOMP_LD_FLAGS) -lm
parallel: all

# Debug build with debug symbols
debug: CFLAGS = -O1 -g $(WFLAGS) -I$(INC_DIR) -DDEBUG=1
debug: clean all

# Clang address sanitizer build
asan: CFLAGS = -fsanitize=address -O1 -g $(WFLAGS) -I$(INC_DIR) -DDEBUG=1
asan: LDFLAGS = -fsanitize=address -lm
asan: clean all

# Clang thread sanitizer build
tsan: CC = mpicc
tsan: CFLAGS = -fno-omit-frame-pointer -fsanitize=thread -O1 -g $(WFLAGS) -I$(INC_DIR) -DDEBUG=1 -DPARALLEL=1
tsan: LDFLAGS = $(BREW_LIBOMP_LD_FLAGS) -fsanitize=thread -lm
tsan: clean all