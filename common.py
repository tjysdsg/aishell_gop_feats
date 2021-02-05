# 声母
INITIALS = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'w', 'x', 'y', 'z', 'zh']

# SSB utterance id to aishell2 utterance id
ssb2utt = {}
with open('ssbutt.txt') as f:
    for line in f:
        utt, ssb = line.replace('\n', '').split('|')
        ssb2utt[ssb] = utt

phone2id = {}
pid2phone = {}
with open('phones.txt') as f:
    for line in f:
        tokens = line.replace('\n', '').split()
        name, pid = tokens
        name = name.replace('_', '')
        pid = int(pid)
        phone2id[name] = pid
        pid2phone[pid] = name
