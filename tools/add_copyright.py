#!/usr/bin/env python3

from os.path import exists
import os
import sys
 
import datetime

title = sys.argv[1] 
date_object = datetime.date.today()

content = None
if os.path.exists(title):
    filename = open(title, 'r+')
    content = filename.read()
    filename.seek(0,0)
else:
    filename = open(title, 'w')

def write_python_template():
    filename.write('# Copyright (c) '+str(date_object.year)+' PeachLab. All Rights Reserved.')
    filename.write('\n# Author : goat.zhou@qq.com (Yang Zhou)')
    filename.write('\n')
    filename.write('\n')

def write_cpp_template():
    filename.write('// Copyright (c) '+str(date_object.year)+' PeachLab. All Rights Reserved.')
    filename.write('\n// Author : goat.zhou@qq.com (Yang Zhou)')
    filename.write('\n')
    filename.write('\n')
 

if title.endswith('.cc') or title.endswith('.cpp') or title.endswith('.c') or title.endswith('h'):
    write_cpp_template()

if title.endswith('.py') or title.endswith('.sh') or title=='BUILD':
    write_python_template()

if content != None: 
    filename.write(content)

filename.close()

exit()
