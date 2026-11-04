// Batch polygon areas and a certified local support enumeration for the MEC.
#include "../src/semicircle_packing/area_kernel.cpp"
#include <array>
#include <limits>

namespace {
struct Disk { double x,y,r; };
struct Circle { double x,y,r; bool ok; };
double boundary_radius(const double* p,int n,double x,double y) {
    double r=0;
    for(int i=0;i<n;i++) {
        double dx=p[3*i]-x,dy=p[3*i+1]-y,t=p[3*i+2];
        double c=std::cos(t),s=std::sin(t);
        r=std::max(r,std::hypot(dx-s,dy+c));
        r=std::max(r,std::hypot(dx+s,dy-c));
        if(dx*c+dy*s>=0) r=std::max(r,std::hypot(dx,dy)+1);
    }
    return r;
}
Circle triple(Disk a,Disk b,Disk c) {
    double bx=b.x-a.x,by=b.y-a.y,cx=c.x-a.x,cy=c.y-a.y;
    double det=bx*cy-by*cx;
    if(std::abs(det)<1e-14)return {0,0,0,false};
    double eb=(bx*bx+by*by+a.r*a.r-b.r*b.r)/2;
    double ec=(cx*cx+cy*cy+a.r*a.r-c.r*c.r)/2;
    double ux=(eb*cy-by*ec)/det,uy=(bx*ec-eb*cx)/det;
    double vx=((b.r-a.r)*cy-by*(c.r-a.r))/det;
    double vy=(bx*(c.r-a.r)-(b.r-a.r)*cx)/det;
    double A=vx*vx+vy*vy-1,B=2*(ux*vx+uy*vy+a.r),C=ux*ux+uy*uy-a.r*a.r;
    double disc=B*B-4*A*C;
    if(disc<0 || std::abs(A)<1e-15)return {0,0,0,false};
    double roots[2]={(-B-std::sqrt(disc))/(2*A),(-B+std::sqrt(disc))/(2*A)};
    for(double r:roots) {
        if(r<std::max({a.r,b.r,c.r}))continue;
        double x=ux+vx*r,y=uy+vy*r;
        // Positive support stress: the center must be inside the center triangle.
        double beta=(x*cy-y*cx)/det,gamma=(bx*y-by*x)/det;
        if(beta < -1e-12 || gamma < -1e-12 || beta+gamma>1+1e-12)continue;
        return {a.x+x,a.y+y,r,true};
    }
    return {0,0,0,false};
}
Circle local_mec(const double* p,int n,double x0,double y0,double window) {
    double r0=boundary_radius(p,n,x0,y0);
    std::vector<Disk> disks;
    for(int i=0;i<n;i++) {
        double x=p[3*i],y=p[3*i+1],t=p[3*i+2],c=std::cos(t),s=std::sin(t);
        if((x-x0)*c+(y-y0)*s>0) {
            if(std::hypot(x-x0,y-y0)+1>=r0-window)disks.push_back({x,y,1});
        } else {
            for(int sign:{-1,1}) {
                double xx=x-sign*s,yy=y+sign*c;
                if(std::hypot(xx-x0,yy-y0)>=r0-window)disks.push_back({xx,yy,0});
            }
        }
    }
    // A covering circle with positive support stress certifies optimality for
    // this packing, provided its radius-one support disks still expose an arc.
    auto covers=[&](Circle q,const std::vector<int>& ids) {
        if(!q.ok)return false;
        for(int index:ids)if(disks[index].r==1) {
            bool exposed=false;
            for(int i=0;i<n;i++)if(p[3*i]==disks[index].x && p[3*i+1]==disks[index].y) {
                exposed=(p[3*i]-q.x)*std::cos(p[3*i+2])+(p[3*i+1]-q.y)*std::sin(p[3*i+2])>=0;
                break;
            }
            if(!exposed)return false;
        }
        for(const auto& d:disks)if(std::hypot(d.x-q.x,d.y-q.y)+d.r>q.r+3e-12)return false;
        return boundary_radius(p,n,q.x,q.y)<=q.r+3e-12;
    };
    for(int i=0;i<(int)disks.size();i++)for(int j=i+1;j<(int)disks.size();j++) {
        auto a=disks[i],b=disks[j];double dx=b.x-a.x,dy=b.y-a.y,d=std::hypot(dx,dy);
        if(d<1e-14)continue;
        double r=(d+a.r+b.r)/2,f=(r-a.r)/d;
        if(r<std::max(a.r,b.r))continue;
        Circle q{a.x+f*dx,a.y+f*dy,r,true};
        if(covers(q,{i,j})) {q.r=std::max(q.r,boundary_radius(p,n,q.x,q.y));return q;}
    }
    for(int i=0;i<(int)disks.size();i++)for(int j=i+1;j<(int)disks.size();j++)for(int k=j+1;k<(int)disks.size();k++) {
        Circle q=triple(disks[i],disks[j],disks[k]);
        if(covers(q,{i,j,k})){q.r=std::max(q.r,boundary_radius(p,n,q.x,q.y));return q;}
    }
    return {x0,y0,r0,false};
}
}
extern "C" void packing_breakout_batch(int count,int n,const double* p,const double* centers,
                                        double window,double* areas,double* circles,int* success) {
    int np=n*(n-1)/2;
    for(int b=0;b<count;b++) {
        const double* q=p+3*n*b;
        packing_polygon_areas(n,q,areas+np*b);
        Circle m=local_mec(q,n,centers[2*b],centers[2*b+1],window);
        circles[3*b]=m.x;circles[3*b+1]=m.y;circles[3*b+2]=m.r;
        success[b]=m.ok?1:0;
    }
}
