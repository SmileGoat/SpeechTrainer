# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

# to do unfinished
# abandon later

import struct

class KaldiEgs(object):
    def __init__

'''
void NnetIo::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<NnetIo>");
  ReadToken(is, binary, &name);
  ReadIndexVector(is, binary, &indexes);
  features.Read(is, binary);
  ExpectToken(is, binary, "</NnetIo>");
}


'''

def expect_token(fd, token):
    size = len(token)
    token_read = fd.read(size).decode()
    return (token == token_read)

def read_int(fd):
    data, = struct.unpack('i', fd.read(4))
    return data

def read_float(fd):
    data, = struct.unpack('f', fd.read(4))
    return data

def read_token(fd):
    token = ''
    while 1:
        char = fd.read(1).decode("latin1")
        if char == '' : break
        if char == ' ' : break
        token += char
    token = token.strip()
    if key == '': return None
    return token 
'''
void ReadToken(std::istream &is, bool binary, std::string *str) {
  KALDI_ASSERT(str != NULL);
  if (!binary) is >> std::ws;  // consume whitespace.
  is >> *str;
  if (is.fail()) {
    KALDI_ERR << "ReadToken, failed to read token at file position "
              << is.tellg();
  }
  if (!isspace(is.peek())) {
    KALDI_ERR << "ReadToken, expected space after token, saw instead "
              << CharToString(static_cast<char>(is.peek()))
              << ", at file position " << is.tellg();
  }
  is.get();  // consume the space.
}

void NnetExample::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Nnet3Eg>");
  ExpectToken(is, binary, "<NumIo>");
  int32 size;
  ReadBasicType(is, binary, &size);
  if (size <= 0 || size > 1000000)
    KALDI_ERR << "Invalid size " << size;
  io.resize(size);
  for (int32 i = 0; i < size; i++)
    io[i].Read(is, binary);
  ExpectToken(is, binary, "</Nnet3Eg>");
}
'''
class NnetIo(object):
    class Index(object):
        def __init__(self):
            self.n = 0 # member-index of minibatch
            self.t = 0 # time-frame
            self.x = 0 # catch-all extra index which maybe used in convolutional setups
        
        def read(self, fd):
            expect_token(fd, '<I1>')
            self.n = read_int(fd)
            self.t = read_int(fd)
            self.x = read_int(fd)


    def __init__(self)：
        self.features = None
        self.name = ""
        self.indexes = []
    
    def read_index_vector(self, fd):
        expect_token(fd, '<I1V>')
        size = read_int(fd)
        for idx in range(0, size, 1):
            Index k
            k.read(fd)
            self.indexes.append(k)

    def read(self, file_obj):
        expect_token(file_obj, '<NnetIo>')
        name = read_token(file_obj)
        #readIndexVector
        
        #features read
        
        expect_token(file_obj, '</NnetIo>')


class NnetExample(object):
    def __init__(self):
        ...
    
    def read(self, file_obj):
        expect_token(file_obj, '<Nnet3Eg>') 
        expect_token(file_obj, '<NumIo>')
        size = 0
        size = read_int(file_obj)
        if (size <= 0 or size > 1000000):
            raise ValueError("invalid size %d" % size)
        list = []
        for i in range(0, size):
            io_read(fd) 
        read_token(file_obj, '</Nnet3Eg>')




