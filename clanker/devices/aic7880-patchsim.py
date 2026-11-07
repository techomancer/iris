#!/usr/bin/env python3
"""Reproduce the Linux aic7xxx driver's firmware patch selection for an AIC-7880.

Usage: aic7880-patchsim.py PATH/TO/aic7xxx_seq.h_shipped

Prints how many of the 880 source instructions survive for each 7880
configuration (revision 0 or >= 1, Ultra, wide) and the critical sections at
their downloaded addresses. Background: docs/dev/aic7880.md section 8.1.
"""
import re,sys
src=open(sys.argv[1]).read()
prog=[int(x,16) for x in re.findall(r'0x([0-9a-f]{2})', src.split('seqprog[] = {')[1].split('};')[0])]
ninstr=len(prog)//4
# patch funcs
funcs={}
for m in re.finditer(r'ahc_patch(\d+)_func\(struct ahc_softc \*ahc\)\n\{\n\treturn \((.*)\);\n\}',src):
    funcs[int(m.group(1))]=m.group(2)
patches=[(int(a),int(b),int(c),int(d)) for a,b,c,d in re.findall(r'\{ ahc_patch(\d+)_func, (\d+), (\d+), (\d+) \}',src)]
cs=[(int(a),int(b)) for a,b in re.findall(r'\{ (\d+), (\d+) \}', src.split('critical_sections[] = {')[1].split('};')[0])]
def mkcfg(rev1=True, ultra=True, wide=True, cachesize=32):
    features=set(['AHC_TARGETMODE','AHC_AUTOPAUSE'])  # TARGETMODE later cleared by ahc_init when AHC_TMODE_ENABLE==0
    features.discard('AHC_TARGETMODE')
    if ultra: features.add('AHC_ULTRA')
    if wide: features.add('AHC_WIDE')
    flags=set(['AHC_PAGESCBS','AHC_INITIATORROLE'])
    bugs=set(['AHC_TMODE_WIDEODD_BUG'])
    if rev1: bugs.add('AHC_PCI_2_1_RETRY_BUG')
    else: bugs.update(['AHC_CACHETHEN_BUG','AHC_PCI_MWI_BUG'])
    return features,flags,bugs,cachesize
def evalf(expr,cfg):
    features,flags,bugs,cachesize=cfg
    e=expr
    def rep(m):
        kind,name=m.group(1),m.group(2)
        s={'features':features,'flags':flags,'bugs':bugs}[kind]
        return 'True' if name in s else 'False'
    e=re.sub(r'\(ahc->(features|flags|bugs) & (AHC_[A-Z0-9_]+)\) != 0',rep,e)
    e=re.sub(r'\(ahc->(features|flags|bugs) & (AHC_[A-Z0-9_]+)\) == 0',lambda m:'False' if m.group(2) in {'features':features,'flags':flags,'bugs':bugs}[m.group(1)] else 'True',e)
    e=e.replace('ahc->pci_cachesize != 0','True' if cachesize else 'False').replace('&&',' and ')
    return eval(e)
def run(cfg):
    results={n:evalf(x,cfg) for n,x in funcs.items()}
    cur=0; skip=0; downloaded=0; dl=[]; 
    cs_tab=[]; begin_set=[False]*len(cs); end_set=[False]*len(cs); cs_count=0; cur_cs=0
    def check(i,cur,skip):
        while cur<len(patches) and i==patches[cur][1]:
            if not results[patches[cur][0]]:
                skip=i+patches[cur][2]; cur+=patches[cur][3]
            else: cur+=1
        return cur,skip,(i>=skip)
    newaddr={}
    for i in range(ninstr):
        cur,skip,ok=check(i,cur,skip)
        if not ok: continue
        while cur_cs<len(cs):
            if cs[cur_cs][1]<=i:
                if begin_set[cs_count] and not end_set[cs_count]:
                    cs_tab[cs_count]=(cs_tab[cs_count][0],downloaded); end_set[cs_count]=True; cs_count+=1
                cur_cs+=1; continue
            if cs[cur_cs][0]<=i and not begin_set[cs_count]:
                cs_tab.append((downloaded,None)); begin_set[cs_count]=True
            break
        newaddr[i]=downloaded; dl.append(i); downloaded+=1
    return results,downloaded,dl,cs_tab,newaddr
for name,cfg in [('7880 revB+ (PCI_2_1_RETRY_BUG), ultra, wide',mkcfg()),('7880 rev0 (CACHETHEN+MWI bugs), ultra, wide, cachesize 32',mkcfg(rev1=False)),('7880 revB+, no REXTVALID (no ULTRA), wide',mkcfg(ultra=False))]:
    results,downloaded,dl,cs_tab,newaddr=run(cfg)
    print('==',name)
    print(' total instrs in seqprog:',ninstr,' downloaded:',downloaded)
    print(' patch funcs true:',sorted(n for n,v in results.items() if v))
    print(' patch funcs false:',sorted(n for n,v in results.items() if not v))
    print(' critical sections (downloaded addrs):',cs_tab)
    print(' source critical sections:',cs)
print()
for n in sorted(funcs): print(n,funcs[n])
