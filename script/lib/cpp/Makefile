CXX = g++
CXXFLAGS= 
EXE   = gen_cnn_code

CXXFLAGS = -std=c++11  
#
GEN_SOURCES += $(shell ls *.cpp)
GEN_OBJS    = $(GEN_SOURCES:.cpp=.o)


all:$(GEN_OBJS)
	$(CXX) -o $(EXE) $(GEN_OBJS) $(LDFLAGS)

%.o :%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -rf $(EXE) *.o
	rm -rf cnn utils
