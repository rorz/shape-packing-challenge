// Exact support-function geometry for unit semicircles, with analytic derivatives.
#include <cmath>
#include <algorithm>
#include <limits>

struct Vec {
    double x,y;
    Vec operator+(Vec b) const { return {x+b.x,y+b.y}; }
    Vec operator-(Vec b) const { return {x-b.x,y-b.y}; }
    Vec operator*(double s) const { return {x*s,y*s}; }
    double dot(Vec b) const { return x*b.x+y*b.y; }
    Vec perp() const { return {-y,x}; }
};
struct Support { double h, hp; };
static Support support(Vec u,Vec n,Vec t) {
    double d=u.dot(n), c=u.dot(t);
    if (d>=0) return {1,0};
    return {std::abs(c),(c>=0?1.0:-1.0)*d};
}
struct Pair {double gap; double jac[6];};
static Pair pair_gap(Vec a,Vec na,Vec ta,Vec b,Vec nb,Vec tb) {
    Vec d=b-a;
    Pair best;best.gap=-std::numeric_limits<double>::infinity();
    auto candidate=[&](Vec u,double phia,double phib,double phiax,double phiay,double phibx,double phiby){
        Support ha=support(u,na,ta),hb=support(u*-1,nb,tb);
        double g=d.dot(u)-ha.h-hb.h;
        if(g<=best.gap) return;
        double gp=d.dot(u.perp())-ha.hp-hb.hp;
        best.gap=g;
        best.jac[0]=-u.x+gp*phiax;
        best.jac[1]=-u.y+gp*phiay;
        best.jac[2]=ha.hp+gp*phia;
        best.jac[3]=u.x+gp*phibx;
        best.jac[4]=u.y+gp*phiby;
        best.jac[5]=hb.hp+gp*phib;
    };
    // A flat side or B flat side supplies the only nonsmooth possible maxima.
    candidate(na*-1,1,0,0,0,0,0);
    candidate(nb,0,1,0,0,0,0);
    // Each smooth arc of the Minkowski difference has radius 0, 1, or 2.
    // Its outward stationary direction points from its center to d.
    for(int ia=-1;ia<=1;++ia) for(int ib=-1;ib<=1;++ib){
        Vec q=d-ta*ia+tb*ib;
        double len=std::hypot(q.x,q.y);
        if(len<1e-14) continue;
        Vec u=q*(1/len),v=u.perp();
        double pa=(na*ia).dot(v)/len; // -(ia*t)' = ia*n
        double pb=(nb*(-ib)).dot(v)/len;
        candidate(u,pa,pb,-v.x/len,-v.y/len,v.x/len,v.y/len);
    }
    return best;
}
extern "C" {
// Keep the two endpoint inequalities separate to preserve both active rows
// when a semicircle is placed radially inward and the endpoints tie.
void packing_boundary_constraints(int n,const double *z,double *g,double *jac) {
    int cols=3*n+1;
    if(jac)std::fill(jac,jac+3*n*cols,0.0);
    for(int i=0;i<n;++i){
        Vec p={z[3*i],z[3*i+1]},normal={std::cos(z[3*i+2]),std::sin(z[3*i+2])},t=normal.perp();
        for(int side=0;side<2;++side){
            double sign=side?1.0:-1.0;Vec far=p+t*sign;
            double r=std::hypot(far.x,far.y);int row=3*i+side;
            g[row]=z[3*n]-r;
            if(jac){
                if(r>1e-14){jac[row*cols+3*i]=-far.x/r;jac[row*cols+3*i+1]=-far.y/r;
                    jac[row*cols+3*i+2]=sign*p.dot(normal)/r;}
                jac[row*cols+3*n]=1;
            }
        }
        int row=3*i+2;double r=std::hypot(p.x,p.y);
        if(p.dot(normal)>=0){
            g[row]=z[3*n]-r-1;
            if(jac){if(r>1e-14){jac[row*cols+3*i]=-p.x/r;jac[row*cols+3*i+1]=-p.y/r;}
                jac[row*cols+3*n]=1;}
        } else {g[row]=1;}
    }
}

// z = [x0,y0,theta0,...,radius]. Every constraint is nonnegative when feasible.
void packing_constraints(int n,const double *z,double *g,double *jac) {
    int cols=3*n+1,rows=n*(n-1)/2+n;
    if(jac) std::fill(jac,jac+rows*cols,0.0);
    int row=0;
    for(int i=0;i<n;++i){
        Vec a={z[3*i],z[3*i+1]},na={std::cos(z[3*i+2]),std::sin(z[3*i+2])},ta=na.perp();
        for(int j=i+1;j<n;++j){
            Vec b={z[3*j],z[3*j+1]},nb={std::cos(z[3*j+2]),std::sin(z[3*j+2])},tb=nb.perp();
            Pair p=pair_gap(a,na,ta,b,nb,tb);
            g[row]=p.gap;
            if(jac) for(int k=0;k<3;++k){jac[row*cols+3*i+k]=p.jac[k];jac[row*cols+3*j+k]=p.jac[k+3];}
            ++row;
        }
    }
    for(int i=0;i<n;++i){
        Vec p={z[3*i],z[3*i+1]},normal={std::cos(z[3*i+2]),std::sin(z[3*i+2])},t=normal.perp();
        double norm=std::hypot(p.x,p.y),radius,dx=0,dy=0,dt=0;
        if(p.dot(normal)>=0){
            radius=norm+1;
            if(norm>1e-14){dx=p.x/norm;dy=p.y/norm;}
        } else {
            double sign=p.dot(t)>=0?1.0:-1.0;
            Vec far=p+t*sign;
            radius=std::hypot(far.x,far.y);
            dx=far.x/radius;dy=far.y/radius;dt=-sign*p.dot(normal)/radius;
        }
        g[row]=z[3*n]-radius;
        if(jac){jac[row*cols+3*i]=-dx;jac[row*cols+3*i+1]=-dy;jac[row*cols+3*i+2]=-dt;jac[row*cols+3*n]=1;}
        ++row;
    }
}
}
