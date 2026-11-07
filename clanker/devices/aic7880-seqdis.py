#!/usr/bin/env python3
"""Disassemble an Adaptec AIC-7xxx sequencer image held as a C byte array.

Usage: aic7880-seqdis.py SEQ_ARRAY.h OUT.dis

Written for the HIM's image; any `UBYTE name[] = { 0x.., ... };` array
works, e.g. a SEQRAM download captured from the monitor and dumped as one.
Encoding and register names follow docs/dev/aic7880.md sections 6 and 7
(register names are the HIM's scratch layout, section 8.2). The listing is
not kept in the repository because the bytes are Adaptec's firmware; run
this to regenerate it.
"""
import re,sys
src=open(sys.argv[1]).read()
body=src[src.index('{')+1:src.index('};')]
b=[int(x,16) for x in re.findall(r'0x([0-9A-Fa-f]{2})',body)]
words=[b[i]|(b[i+1]<<8)|(b[i+2]<<16)|(b[i+3]<<24) for i in range(0,len(b),4)]
# LE register names (chip-native offsets)
R={0x00:'SCSISEQ',0x01:'SXFRCTL0',0x02:'SXFRCTL1',0x03:'SCSISIG',0x04:'SCSIRATE',0x05:'SCSIID',0x06:'SCSIDATL',0x07:'SCSIDATH',
0x08:'STCNT0',0x09:'STCNT1',0x0a:'STCNT2',0x0b:'SSTAT0/CLRSINT0',0x0c:'SSTAT1/CLRSINT1',0x0d:'SSTAT2',0x0e:'SSTAT3',0x0f:'SCSITEST',
0x10:'SIMODE0',0x11:'SIMODE1',0x12:'SCSIBUSL',0x13:'SCSIBUSH',0x14:'SHADDR0',0x15:'SHADDR1',0x16:'SHADDR2',0x17:'SHADDR3',
0x19:'SELID',0x1d:'BRDCTL',0x1e:'SEEPROM',0x1f:'SBLKCTL',
0x20:'XFER_OPTION+0',0x21:'XFER_OPTION+1',0x22:'XFER_OPTION+2',0x23:'XFER_OPTION+3',0x24:'XFER_OPTION+4',0x25:'XFER_OPTION+5',0x26:'XFER_OPTION+6',0x27:'XFER_OPTION+7',
0x28:'XFER_OPTION+8',0x29:'XFER_OPTION+9',0x2a:'XFER_OPTION+10',0x2b:'XFER_OPTION+11',0x2c:'XFER_OPTION+12',0x2d:'XFER_OPTION+13',0x2e:'XFER_OPTION+14',0x2f:'XFER_OPTION+15',
0x30:'FAST20_LOW',0x31:'FAST20_HIGH',0x32:'DISCON_OPTION_L',0x33:'DISCON_OPTION_H',
0x3a:'WAITING_SCB',0x3b:'ACTIVE_SCB',0x3c:'PASS_TO_DRIVER',
0x3d:'SCB_PTR_ARRAY+0',0x3e:'SCB_PTR_ARRAY+1',0x3f:'SCB_PTR_ARRAY+2',0x40:'SCB_PTR_ARRAY+3',
0x41:'QIN_CNT',0x42:'QIN_PTR_ARRAY+0',0x43:'QIN_PTR_ARRAY+1',0x44:'QIN_PTR_ARRAY+2',0x45:'QIN_PTR_ARRAY+3',
0x56:'QOUT_PTR_ARRAY+0',0x57:'QOUT_PTR_ARRAY+1',0x58:'QOUT_PTR_ARRAY+2',0x59:'QOUT_PTR_ARRAY+3',
0x5a:'BUSY_PTR_ARRAY+0',0x5b:'BUSY_PTR_ARRAY+1',0x5c:'BUSY_PTR_ARRAY+2',0x5d:'BUSY_PTR_ARRAY+3',
0x60:'SEQCTL',0x61:'SEQRAM',0x62:'SEQADDR0',0x63:'SEQADDR1',0x64:'ACCUM',0x65:'SINDEX',0x66:'DINDEX',0x67:'BRKADDR0',0x68:'BRKADDR1',
0x69:'ALLONES',0x6a:'ALLZEROS/NONE',0x6b:'FLAGS',0x6c:'SINDIR',0x6d:'DINDIR',0x6e:'FUNCTION1',0x6f:'STACK',
0x84:'COMMAND',0x85:'LATTIME',0x86:'PCISTATUS',0x87:'HCNTRL',0x88:'HADDR0',0x89:'HADDR1',0x8a:'HADDR2',0x8b:'HADDR3',
0x8c:'HCNT0',0x8d:'HCNT1',0x8e:'HCNT2',0x90:'SCBPTR',0x91:'INTSTAT',0x92:'CLRINT/ERROR',0x93:'DFCNTRL',0x94:'DFSTATUS',
0x95:'DFWADDR0',0x96:'DFWADDR1',0x97:'DFRADDR0',0x98:'DFRADDR1',0x99:'DFDAT',0x9a:'SCBCNT',0x9b:'QINFIFO',0x9c:'QINCNT',0x9d:'QOUTFIFO',0x9e:'QOUTCNT',0x9f:'SFUNCT'}
for i in range(0x46,0x56): R[i]='NEXT_SCB_ARRAY+%d'%(i-0x46)
for i in range(0xa0,0xc0): R[i]='SCB%02d'%(i-0xa0)
def rn(x): return R.get(x,'r%02x'%x)
OPS={0:'or',1:'and',2:'xor',3:'add',4:'adc',5:'rol',6:'bmov',8:'jmp',9:'jc',10:'jnc',11:'call',12:'jne',13:'jnz',14:'je',15:'jz'}
out=[]
for pc,w in enumerate(words):
    imm=w&0xff; src=(w>>8)&0xff; op=(w>>25)&0xf; hi=(w>>29)
    if op>=8:
        addr=(w>>16)&0x1ff
        s='%-4s %s, 0x%02x -> 0x%03x' % (OPS[op], rn(src), imm, addr)
        if op==8: s='jmp -> 0x%03x   (src %s imm 0x%02x)'%(addr,rn(src),imm)
        if op==11: s='call -> 0x%03x  (src %s imm 0x%02x)'%(addr,rn(src),imm)
    else:
        dst=(w>>16)&0xff; ret=(w>>24)&1
        s='%-4s %s = %s %s 0x%02x%s' % (OPS.get(op,'op%d'%op), rn(dst), rn(src), {0:'|',1:'&',2:'^',3:'+',4:'+c+',5:'rol',6:'bmov'}[op], imm, '  ;ret' if ret else '')
    out.append('%03x: %08x  %s%s' % (pc,w,s,'  ;;hi=%d'%hi if hi else ''))
open(sys.argv[2],'w').write('\n'.join(out)+'\n')
print(len(words),'instructions')
