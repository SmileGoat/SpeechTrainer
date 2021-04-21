#!/usr/bin/python3
# coding=utf-8

import sys

def Usage():
    reminder = "Usage: {app} input_score output_score\n".format(app=sys.argv[0])
    print(reminder)
    sys.exit(-1)

def main(score_file, writer_file):
    writer = open(writer_file, 'w')
    with open(score_file, 'r') as f:
        for line in f:
            flag = "target"
            line = line.strip()
            spk, utt_id, _ = line.split()
            match_spk = utt_id.split('_')[0]
            if spk != match_spk:
                flag = "nontarget"
            write_line = line + " " + flag + "\n"
            writer.write(write_line)
    writer.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        Usage()
    score_file = sys.argv[1]
    writer_file = sys.argv[2]
    main(score_file, writer_file)


