"""Editable, PDF-only BiTFI methods diagram; schematic curves, not measurements.

Step 1: independent bidirectional initialization. Step 2: synchronous bidirectional refinement passes using a snapshot fixed
within each pass, other local variables, calendar and training sites.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import prism_style as ps
from experiment_paths import experiment_path

COLORS=['#CD3658','#DD8093','#009FBD','#00A579','#F39B2D']
INK='#15375B'; GRAY='#777777'; GAP='#EEEEEE'

def main():
    selection=experiment_path('refinement_validation/selection.json','03_result/refinement_validation_20260926/selection.json')
    depth=json.loads(selection.read_text())["selected_refinements"] if selection.exists() else 1
    ps.setup()
    plt.rcParams.update({'pdf.fonttype':42,'font.size':8,'font.family':'sans-serif'})
    fig=plt.figure(figsize=(6.8,5.9));ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,180),ylim=(160,0));ax.axis('off')
    def text(x,y,s,size=8,weight='normal',color=INK,ha='left'):
        ax.text(x,y,s,fontsize=max(7.8,size),fontweight=weight,color=color,ha=ha,va='center',linespacing=1.35)
    def line(x,y,c=INK,lw=.8,ls='-'):ax.plot(x,y,color=c,lw=lw,ls=ls)
    def arrow(x1,y1,x2,y2):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=8,color=INK,lw=.8))
    def box(x,y,w,h,label,face='white',size=7.4):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=face,edgecolor=INK,lw=.7))
        text(x+w/2,y+h/2,label,size,ha='center')
    def series(x0,x1,top,spacing,mode):
        t=np.linspace(0,1,301);g=(t>=.4)&(t<=.6);xx=x0+(x1-x0)*t
        ax.add_patch(Rectangle((x0+.4*(x1-x0),top-5),.2*(x1-x0),4*spacing+10,fc=GAP,ec='none'))
        for j,c in enumerate(COLORS):
            z=np.sin(6*np.pi*t+.25*j)+.23*np.sin(16*np.pi*t+.5*j)
            if j==2:z=-z+.15*np.sin(31*np.pi*t)
            if j==3:z=.5*z+.12*np.sin(37*np.pi*t)
            if j==4:z=np.maximum(0,np.sin(6*np.pi*t-.3)) *1.9-.8
            yy=top+j*spacing-2*z
            observed=yy.copy();observed[g]=np.nan;line(xx,observed,c,1.1)
            if mode!='missing':
                m=yy.copy();m[~g]=np.nan;line(xx,m,c,1.3,'--' if mode=='initial' else '-')
        return x0+.4*(x1-x0),x0+.6*(x1-x0)
    # Three aligned columns echo the author's graphical abstract.
    for x,w,face,title in [(3,47,'#E6E6E6','A  Incomplete records'),(57,66,'#DCECF6','B  Bidirectional imputation'),(130,47,'#C6E0F1','C  Reconstructed records')]:
        ax.add_patch(Rectangle((x,3),w,9,fc=face,ec='none'))
        text(x+w/2,7.5,title,8,weight='bold',ha='center')
    labels=[r'$T_{\mathrm{in}}$',r'$T_{\mathrm{out}}$','RH',r'CO$_2$','Rad']
    for x in [4,131]:
        for j,label in enumerate(labels):text(x,30+11*j,label,8,color=COLORS[j])
    series(15,49,30,11,'missing');series(142,176,30,11,'final')
    for x0,x1 in [(15,49),(142,176)]:
        text((x0+x1)/2,20,'Gap' if x0==15 else 'Imputed',7.5,ha='center',color=GRAY)
        arrow(x0,84,x1,84);text((x0+x1)/2,89,'Time',7.5,ha='center')
    # Step 1 supplies independent initial estimates; no covariates enter this step.
    arrow(50,37,58,37)
    text(90,19,'Step 1',8.8,weight='bold',ha='center')
    text(90,25,'Bidirectional initialization',7.8,ha='center')
    text(73,31,'Before gap',6.9,ha='center')
    text(108,31,'After gap, reversed',6.9,ha='center')
    box(59,35,28,10,'Forward prediction')
    box(94,35,28,10,'Backward prediction')
    line([73,73,90],[45,49,49]);line([108,108,90],[45,49,49])
    arrow(90,49,90,52)
    box(65,52,50,10,'Align + weighted fusion','#F4F7FA')
    arrow(90,62,90,75)
    text(94,68,'Step 1 estimates',7)
    # A single fixed snapshot is shared by all Step 2 calls.
    box(57,75,66,16,'Fixed snapshot within each pass\nOther sensors: observed or imputed'+('\nRefreshed between passes' if depth>1 else ''),'#EDF4F8',6.9)
    text(90,95,"Target's own gap estimates excluded",6.7,ha='center',color=GRAY)
    text(90,103,'Step 2',8.8,weight='bold',ha='center')
    text(90,109,'Bidirectional covariate refinement',7.5,ha='center')
    # Keep the snapshot branch outside the step heading.
    line([57,55,55,73],[83,83,114,114])
    line([123,125,125,108],[83,83,114,114])
    arrow(73,114,73,117);arrow(108,114,108,117)
    box(59,117,28,12,'Forward\n+ covariates')
    box(94,117,28,12,'Backward\n+ covariates')
    line([73,73,90],[129,133,133]);line([108,108,90],[129,133,133])
    arrow(90,133,90,136)
    box(65,136,50,10,'Align + weighted fusion','#F4F7FA')
    line([115,128,128],[141,141,49]);arrow(128,49,133,49)
    # Local and external sources are grouped beside the fixed snapshot.
    line([3,49],[99,99],c='#B9C6CE',lw=.6)
    text(3,105,'Additional inputs to Step 2',8,weight='bold')
    text(3,114,'Other local sensor records',7.2)
    text(3,123,'Calendar: hour and day of year',7.2)
    text(3,132,'Same-variable records from',7.2)
    text(3,138,'up to 3 training greenhouses',7.2)
    line([49,52,52],[119,119,83]);arrow(52,83,57,83)
    # Explain directional handling and the non-iterative refinement explicitly.
    line([133,177],[99,99],c='#B9C6CE',lw=.6)
    text(133,105,'In both steps',8,weight='bold')
    text(133,114,'Restore backward time order',7.1)
    text(133,123,'Fuse by distance to gap edges',7.1)
    text(133,136,f'{depth} refinement '+('pass' if depth==1 else 'passes'),7.2,weight='bold')
    if depth>1:
        line([115,126,126],[141,141,83]);arrow(126,83,123,83)
        ax.text(127,110,'Repeat',fontsize=7.8,color=INK,ha='center',va='center',rotation=90)
    text(133,143,'Observed values are retained',7.1)
    line([3,177],[152,152],c='#B9C6CE',lw=.6)
    text(90,157,'Shared frozen backbone in both steps  ·  No parameter updates',7.6,ha='center')
    ps.OUT.mkdir(exist_ok=True)
    path=ps.OUT/'Figure2_BiTFI_framework.pdf';fig.savefig(path);plt.close(fig);print(path)

if __name__=='__main__':main()
