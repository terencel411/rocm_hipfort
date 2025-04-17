.PHONY: build run clean

# compiler
HIPFC = hipfc

# compiler flags
VERBOSE_FLAG = -v
OFFLOAD_ARCH_FLAG = --offload-arch=gfx90a
CFLAGS = $(VERBOSE_FLAG) $(OFFLOAD_ARCH_FLAG)

# source files
CPP_SRC = hip_implementation.cpp
FORT_SRC = main.f03
OUTPUT = main

build: $(OUTPUT)

$(OUTPUT):
	$(HIPFC) $(CFLAGS) $(CPP_SRC) $(FORT_SRC) -o $(OUTPUT)

run: $(OUTPUT)
	./$(OUTPUT)

clean:
	rm -f $(OUTPUT) *.o *.mod