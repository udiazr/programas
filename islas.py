##UADR this program is released without any warranty implicit or explicit to the public domain

import sys
#from PyQt5.QtCore import Qt
#from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QGridLayout, QWidget, QCheckBox, QGridLayout, QGroupBox,QMenu, QPushButton, QRadioButton, QVBoxLayout, QWidget, QSlider
#from PyQt5.QtGui import QPixmap, QImage 
#import PyQt5.QtGui
import numpy as np
import cv2
import pyopencl as cl


filename=sys.argv[1]
img0=255-cv2.imread(filename,cv2.IMREAD_GRAYSCALE)


h0,w0=img0.shape
dh=30
h=int(h0+dh*2)
w=int(w0+dh*2)
img=np.zeros((h,w),dtype=np.uint8)
img[dh:h0+dh,dh:w0+dh]=img0

print(h,w)

ctx=cl.create_some_context()
queue=cl.CommandQueue(ctx)
mf=cl.mem_flags

program="""
__kernel void k0(__global unsigned char *img,__global unsigned *X){
int i=get_global_id(0),j=get_global_id(1),n=get_global_size(0);
int th=100;
if(img[i+j*n]>th)
X[i+j*n]=i+j*n;
else
X[i+j*n]=0;
}

__kernel void k1(__global unsigned *X){
int i=get_global_id(0),j=get_global_id(1),n=get_global_size(0),m=get_global_size(1);
unsigned v=X[i+j*n];
if(X[i+j*n]>0){
int ii,jj;

ii=(i+1)%n,jj=j;
if(X[ii+jj*n]>0)
v=min(X[ii+jj*n],v);

ii=(i-1+n)%n,jj=j;
if(X[ii+jj*n]>0)
v=min(X[ii+jj*n],v);

ii=i,jj=(j+1)%m;
if(X[ii+jj*n]>0)
v=min(X[ii+jj*n],v);

ii=i,jj=(j-1+m)%m;
if(X[ii+jj*n]>0)
v=min(X[ii+jj*n],v);


ii=(i+1)%n,jj=(j+1)%m;
if(X[ii+jj*n]>0)
v=min(X[ii+jj*n],v);

ii=(i+1)%n,jj=(j-1+m)%m;
if(X[ii+jj*n]>0)
v=min(X[ii+jj*n],v);


ii=(i+n-1)%n,jj=(j+1)%m;
if(X[ii+jj*n]>0)
v=min(X[ii+jj*n],v);

ii=(i+n-1)%n,jj=(j-1+m)%m;
if(X[ii+jj*n]>0)
v=min(X[ii+jj*n],v);


X[i+j*n]=v;
}

}


__kernel void k2(__global unsigned *X,int m){
int i=get_global_id(0),j=get_global_id(1),n=get_global_size(0);
for(int j=0;j<m;j++){
unsigned v=X[i+j*n];
if(X[i+j*n]>0){
int ii,jj;

ii=(i+1)%n,jj=j;
if(X[ii+jj*n]>0)
v=min(X[ii+jj*n],v);

ii=(i-1+n)%n,jj=j;
if(X[ii+jj*n]>0)
v=min(X[ii+jj*n],v);

ii=i,jj=(j+1)%m;
if(X[ii+jj*n]>0)
v=min(X[ii+jj*n],v);

ii=i,jj=(j-1+m)%m;
if(X[ii+jj*n]>0)
v=min(X[ii+jj*n],v);


ii=(i+1)%n,jj=(j+1)%m;
if(X[ii+jj*n]>0)
v=min(X[ii+jj*n],v);

ii=(i+1)%n,jj=(j-1+m)%m;
if(X[ii+jj*n]>0)
v=min(X[ii+jj*n],v);


ii=(i+n-1)%n,jj=(j+1)%m;
if(X[ii+jj*n]>0)
v=min(X[ii+jj*n],v);

ii=(i+n-1)%n,jj=(j-1+m)%m;
if(X[ii+jj*n]>0)
v=min(X[ii+jj*n],v);


X[i+j*n]=v;
}

}
}

__kernel void k3(__global unsigned *X,__global unsigned *Y,int m){
int i=get_global_id(0),j=get_global_id(1),n=get_global_size(0);
for(int j=0;j<m;j++)
Y[X[i+j*n]]=X[i+j*n];
}


__kernel void k4(__global unsigned *Y,int m){
int i=get_global_id(0),j=get_global_id(1),n=get_global_size(0);
for(int j=0;j<m;j++)
  if(Y[i+j*n]>0)
    Y[atomic_inc(Y)+1]=Y[i+j*n];
}

__kernel void k5(__global unsigned *X,__global unsigned char *Y, int m){
int i=get_global_id(0),j=get_global_id(1),n=get_global_size(0);
for(int j=0;j<m;j++){
if(X[i+j*n]>0)
  Y[i+j*n]=1;
else
  Y[i+j*n]=0;
}
}

__kernel void k6(__global unsigned *X,__global unsigned char *Y, int m){
int i=get_global_id(0),j=get_global_id(1),n=get_global_size(0);
for(int j=0;j<m;j++){
int v=X[i+j*n]*Y[i+j*n];
if(v>0){
int ii,jj;
ii=(i+1)%n,jj=j;
if(X[ii+jj*n]==0)
X[ii+jj*n]=v;

ii=(i+n-1)%n,jj=j;
if(X[ii+jj*n]==0)
X[ii+jj*n]=v;


ii=i,jj=(j+1)%m;
if(X[ii+jj*n]==0)
X[ii+jj*n]=v;

ii=i,jj=(j+m-1)%m;
if(X[ii+jj*n]==0)
X[ii+jj*n]=v;

}}

}


__kernel void k7(__global unsigned *X,__global unsigned char *Y, int m){
int i=get_global_id(0),j=get_global_id(1),n=get_global_size(0);
for(int j=0;j<m;j++){
unsigned v=X[i+j*n];
unsigned v1=v;
if(1){
int ii,jj;

ii=(i+1)%n,jj=j;
v=min(X[ii+jj*n],v);
v1=max(X[ii+jj*n],v1);

ii=(i+n-1)%n,jj=j;
v=min(X[ii+jj*n],v);
v1=max(X[ii+jj*n],v1);


ii=i,jj=(j+1)%m;
v=min(X[ii+jj*n],v);
v1=max(X[ii+jj*n],v1);

ii=i,jj=(j+m-1)%m;
v=min(X[ii+jj*n],v);
v1=max(X[ii+jj*n],v1);
}

if(v1>v)
Y[i+j*n]=0xff;
else
Y[i+j*n]=0;
}


}

"""

prg=cl.Program(ctx,program).build()
to_gpu= lambda x : cl.Buffer(ctx,mf.READ_WRITE|mf.COPY_HOST_PTR,hostbuf=x)


thr=img.copy()
X=np.zeros((h,w),dtype=np.uint32)
img_g=to_gpu(img)
X_g=to_gpu(X)
Y_g=to_gpu((X*0).astype(np.uint32))
k0=prg.k0
k0(queue,[h,w],None,img_g,X_g)

k1=prg.k1
k2=prg.k2
k3=prg.k3
k5=prg.k5
k6=prg.k6
k7=prg.k7

k4=prg.k4
k2.set_scalar_arg_dtypes([None,np.int32])
k3.set_scalar_arg_dtypes([None,None,np.int32])
k4.set_scalar_arg_dtypes([None,np.int32])
#k5.set_scalar_arg_dtypes([None,np.int32])
k5.set_scalar_arg_dtypes([None,None,np.int32])

k6.set_scalar_arg_dtypes([None,None,np.int32])
k7.set_scalar_arg_dtypes([None,None,np.int32])


cl.enqueue_copy(queue,thr,img_g)
cl.enqueue_copy(queue,X,X_g)

if 1:
    for i in range(1000):
        k2(queue,[w],None,X_g,h)
    cl.enqueue_copy(queue,X,X_g)
    cv2.imshow("hola",((X>0)*255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.imshow("hola",X.astype(np.uint8))
    cv2.waitKey(0)


k3(queue,[w],None,X_g,Y_g,h)
Z=np.zeros((h,w,4),dtype=np.uint8)
cl.enqueue_copy(queue,Z,X_g)
Y=np.zeros(X.shape,dtype=np.uint8)


cv2.imshow("hola",Z)
cv2.waitKey(0)

for k in range(int(sys.argv[2])):
    for j in range(2):
        k5(queue,[w],None,X_g,Y_g,h)
        k6(queue,[w],None,X_g,Y_g,h)        
    cl.enqueue_copy(queue,Z,X_g)
    cv2.imshow(f"iteracion:{k*2}",Z)
    cv2.waitKey(0)
Z[:,:,3]=255
cv2.imwrite("salida.png",Z)

cv2.imshow("Imagenfinal",Z)
cv2.waitKey(0)



k7(queue,[w],None,X_g,Y_g,h)        

cl.enqueue_copy(queue,Y,Y_g)
cv2.imwrite("salidaorilla.png",255-Y)

cv2.imshow("Orilla",255-Y)
cv2.waitKey(0)

# k4(queue,[w],None,Y_g,h)

# cl.enqueue_copy(queue,Y,Y_g)
# Y=Y.ravel()
# print(Y[1:Y[0]+1])
# #print(YY)
