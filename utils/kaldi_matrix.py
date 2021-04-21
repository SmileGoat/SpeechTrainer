# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)


import numpy  as np
import os
import struct
import sys

class Matrix(object):
    def __init__(self, ark_file=None):
        if (ark_file == None) :
            return
        self.fd = open(ark_file, 'wb')

    def IsToken(self, key:str):
       if not isinstance(key, str) or key.isspace():
           return False
       return ' ' not in key

    def _read_mat_binary(self, fd):
        # Data type
        binary = fd.read(2).decode()
        assert binary == '\0B' 

        header = fd.read(3).decode()
        sample_size = 0
        # 'CM', 'CM2', 'CM3' are possible values,
        if header == 'FM ': sample_size = 4 # floats
        elif header == 'DM ': sample_size = 8 # doubles
        assert(sample_size > 0)
        # Dimensions
        s1, rows, s2, cols = np.frombuffer(fd.read(10), dtype='int8,int32,int8,int32', count=1)[0]
        # Read whole matrix
        buf = fd.read(rows * cols * sample_size)
        if sample_size == 4 : vec = np.frombuffer(buf, dtype='float32')
        elif sample_size == 8 : vec = np.frombuffer(buf, dtype='float64')
        mat = np.reshape(vec,(rows,cols))
        return mat

    def read(self, file, pos):
        pos = int(pos)
        fd = open(file, 'rb')
        fd.seek(pos, 0)
        return self._read_mat_binary(fd)
    
    def read_mat_scp(self, scp):
        fd = open(scp, 'r')
        try:
            for line in fd:
                (key, ark_file) = line.split()
                file_, pos = ark_file.split(':')
                data = self.read(file_, pos) 
                yield key, data
        finally:
            fd.close()

    def write(self, key='', matrix = None):

        assert self.IsToken(key) , "the key must be a token" 
        self.fd.write((key+' ').encode("latin1"))
        self.fd.write('\0B'.encode())
        # write as binary
        if matrix.dtype == 'float32':
            self.fd.write('FM '.encode())
        elif matrix.dtype == 'float64':
            self.fd.write('DM '.encode())
        rows, cols = matrix.shape
        #write shape
        self.fd.write('\04'.encode())
        self.fd.write(struct.pack('I', rows))
        self.fd.write('\04'.encode())
        self.fd.write(struct.pack('I', cols))
        self.fd.write(matrix.tobytes())
    
    def write_vector(self, key = '', vector = None):
        assert self.IsToken(key), "the key must be a token"
        self.fd.write((key+' ').encode("latin1")) # ark-files have keys (utterance-id),
        self.fd.write('\0B'.encode()) # we write binary!
        # Data-type,
        if vector.dtype == 'float32': self.fd.write('FV '.encode())
        elif vector.dtype == 'float64': self.fd.write('DV '.encode())
        # Dim,
        self.fd.write('\04'.encode())
        self.fd.write(struct.pack(np.dtype('uint32').char, vector.shape[0])) # dim
        # Data,
        self.fd.write(vector.tobytes())

    def close(self):
        self.fd.close()

if __name__ == '__main__':
    test_matrix = np.ones([10, 10], dtype=np.float64)
    test_matrix[3,3]= 3.45675434
    matrix_writer = Matrix('./test.ark')
    key_test = 'test'
    for i in range(10):
        key = key_test + '_' + str(i)
        matrix_writer.write(key, test_matrix)

    matrix_writer = Matrix()
    for key, data in matrix_writer.read_mat_scp('./test_matrix.scp'):
        print(key, data)

    vector_writer = Matrix('./test_vec.ark')
    test_vector = np.array([2.3,3,4,6.0,48.9], dtype=np.float64)
    key_test = 'test_vec'
    for i in range(11):
        key = key_test + '_' + str(i)
        vector_writer.write_vector(key, test_vector)


