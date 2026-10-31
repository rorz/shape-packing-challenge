#include <algorithm>
#include <cmath>
#include <vector>

namespace {
constexpr double PI = 3.141592653589793238462643383279502884;
struct V {
    double x, y;
    V operator+(V b) const { return {x+b.x,y+b.y}; }
    V operator-(V b) const { return {x-b.x,y-b.y}; }
    V operator*(double b) const { return {x*b,y*b}; }
};
double dot(V a,V b) { return a.x*b.x+a.y*b.y; }
double cross(V a,V b) { return a.x*b.y-a.y*b.x; }
struct S { V c,n,t; double theta; };
S shape(const double* p,V origin) {
    double c=std::cos(p[2]),s=std::sin(p[2]);
    return {{p[0]-origin.x,p[1]-origin.y},{c,s},{-s,c},p[2]};
}
bool inside(V p,const S& s) {
    V d=p-s.c;
    return dot(d,d)<=1.0+1e-13 && dot(d,s.n)>=-1e-13;
}
void arc_cut(std::vector<double>& cuts,double angle,double start) {
    double d=std::fmod(angle-start,2*PI);
    if(d<0) d+=2*PI;
    if(d>1e-13 && d<PI-1e-13) cuts.push_back(start+d);
}
double chord_deficit_integral(double angle) {
    constexpr double step=PI/4095;
    double whole=std::floor(angle/step),u=angle-whole*step;
    return whole*step*step*step/12+step*u*u/4-u*u*u/6;
}
double arc_integral(const S& a,const S& b,double* grad=nullptr,double* deficit=nullptr) {
    double start=a.theta-PI/2;
    std::vector<double> cuts{start,start+PI};
    V dc=b.c-a.c;
    double d=std::hypot(dc.x,dc.y);
    if(d>1e-14 && d<2) {
        double angle=std::atan2(dc.y,dc.x),off=std::acos(d/2);
        arc_cut(cuts,angle-off,start); arc_cut(cuts,angle+off,start);
    }
    double k=-dot(a.c-b.c,b.n);
    if(k>-1 && k<1) {
        double off=std::acos(k);
        arc_cut(cuts,b.theta-off,start);arc_cut(cuts,b.theta+off,start);
    }
    std::sort(cuts.begin(),cuts.end());
    double area=0;
    for(size_t i=1;i<cuts.size();++i) {
        double lo=cuts[i-1],hi=cuts[i],mid=(lo+hi)/2;
        if(inside(a.c+V{std::cos(mid),std::sin(mid)},b)) {
            double gx=std::sin(hi)-std::sin(lo),gy=std::cos(lo)-std::cos(hi);
            area+=a.c.x*gx+a.c.y*gy+hi-lo;
            if(grad){grad[0]+=gx;grad[1]+=gy;}
            if(deficit)*deficit+=chord_deficit_integral(hi-start)-chord_deficit_integral(lo-start);
        }
    }
    return area/2;
}
// Derivative of the first-order chord correction, including moving contact
// endpoints and the rotation of the arc's own polygon sampling grid.
struct ChordCut { double angle; double jac[6]; };
double chord_deficit(double angle) {
    constexpr double step=PI/4095;
    double u=angle-std::floor(angle/step)*step;
    return 0.5*u*(step-u);
}
void arc_deficit_gradient(const S& a,const S& b,double* ga,double* gb) {
    double start=a.theta-PI/2;
    ChordCut first{},last{};first.angle=start;last.angle=start+PI;
    first.jac[2]=last.jac[2]=1;
    std::vector<ChordCut> cuts{first,last};
    auto add=[&](double angle,const double* derivatives){
        double offset=std::fmod(angle-start,2*PI);if(offset<0)offset+=2*PI;
        if(offset<=1e-13 || offset>=PI-1e-13)return;
        ChordCut cut{};cut.angle=start+offset;
        std::copy(derivatives,derivatives+6,cut.jac);cuts.push_back(cut);
    };
    V dc=b.c-a.c;double distance=std::hypot(dc.x,dc.y);
    if(distance>1e-14 && distance<2){
        double angle=std::atan2(dc.y,dc.x),off=std::acos(distance/2);
        V unit=dc*(1/distance),perp{-unit.y,unit.x};
        double height=std::sqrt(std::max(1e-30,1-distance*distance/4));
        for(int sign : {-1,1}){
            V gradient=perp*(1/distance)-unit*(sign/(2*height));
            double derivative[6]={-gradient.x,-gradient.y,0,gradient.x,gradient.y,0};
            add(angle+sign*off,derivative);
        }
    }
    double k=dot(dc,b.n);
    if(k>-1 && k<1){
        double off=std::acos(k),height=std::sqrt(1-k*k);
        for(int sign : {-1,1}){
            V gradient=b.n*(sign/height);
            double derivative[6]={gradient.x,gradient.y,0,-gradient.x,-gradient.y,
                1-sign*dot(dc,b.t)/height};
            add(b.theta+sign*off,derivative);
        }
    }
    std::sort(cuts.begin(),cuts.end(),[](const ChordCut& x,const ChordCut& y){return x.angle<y.angle;});
    double derivative[6]={};
    for(size_t i=1;i<cuts.size();++i){
        const ChordCut &lo=cuts[i-1],&hi=cuts[i];double mid=(lo.angle+hi.angle)/2;
        if(!inside(a.c+V{std::cos(mid),std::sin(mid)},b))continue;
        double lower=chord_deficit(lo.angle-start),upper=chord_deficit(hi.angle-start);
        for(int k=0;k<6;++k)derivative[k]+=upper*hi.jac[k]-lower*lo.jac[k];
        derivative[2]-=upper-lower;
    }
    for(int k=0;k<3;++k){ga[k]-=derivative[k];gb[k]-=derivative[k+3];}
}

double line_integral(const S& a,const S& b,double* grad=nullptr) {
    V p=a.c+a.t,v=a.t*(-2),d=p-b.c;
    std::vector<double> cuts{0,1};
    auto cut=[&](double t){ if(t>1e-13 && t<1-1e-13) cuts.push_back(t); };
    double bv=dot(d,v),disc=bv*bv-dot(v,v)*(dot(d,d)-1);
    if(disc>0) { double h=std::sqrt(disc);cut((-bv-h)/4);cut((-bv+h)/4); }
    double den=dot(v,b.n);
    if(std::abs(den)>1e-14) cut(-dot(d,b.n)/den);
    std::sort(cuts.begin(),cuts.end());
    double length=0;
    // A shared flat boundary on the same side has two possible one-sided
    // derivatives. Half its flux gives a translation-invariant subgradient.
    double grad_weight=(dot(a.n,b.n)>1-1e-13
        && std::abs(cross(a.n,b.n))<1e-12
        && std::abs(dot(a.c-b.c,a.n))<1e-12)?0.5:1.0;
    for(size_t i=1;i<cuts.size();++i) {
        double lo=cuts[i-1],hi=cuts[i];
        if(inside(p+v*((lo+hi)/2),b)) {
            length+=hi-lo;
            if(grad) grad[2]+=grad_weight*((2*hi-2*hi*hi)-(2*lo-2*lo*lo));
        }
    }
    if(grad){grad[0]-=grad_weight*a.n.x*2*length;grad[1]-=grad_weight*a.n.y*2*length;}
    return cross(p,v)*length/2;
}
double area(const double* ap,const double* bp,double* ga=nullptr,double* gb=nullptr,double* deficit=nullptr) {
    if(deficit)*deficit=0;
    if(ga) std::fill(ga,ga+3,0.0);
    if(gb) std::fill(gb,gb+3,0.0);
    V origin{ap[0],ap[1]};
    S a=shape(ap,origin),b=shape(bp,origin);
    double distance_sq=dot(b.c,b.c);
    if(distance_sq>=4) return 0;
    // Coincident disks share a curved boundary. Integrating both boundaries
    // would count that arc twice; their intersection is exactly a sector.
    // Below 1e-10 separation this limit has error at most 2e-10 and avoids
    // loss of precision when classifying almost identical circular arcs.
    if(distance_sq<1e-20) {
        double delta=std::abs(std::remainder(ap[2]-bp[2],2*PI));
        if(ga || gb) {
            // At coincident boundaries an ordinary derivative need not exist.
            // Symmetric differences supply a useful consistent subgradient.
            constexpr double h=1e-6;
            double aa[3]={ap[0],ap[1],ap[2]},bb[3]={bp[0],bp[1],bp[2]};
            for(int k=0;k<3;++k){
                if(ga){aa[k]+=h;double plus=area(aa,bb);aa[k]-=2*h;
                    double minus=area(aa,bb);aa[k]+=h;ga[k]=(plus-minus)/(2*h);}
                if(gb){bb[k]+=h;double plus=area(aa,bb);bb[k]-=2*h;
                    double minus=area(aa,bb);bb[k]+=h;gb[k]=(plus-minus)/(2*h);}
            }
        }
        return (PI-delta)/2;
    }
    // Identical diameter segments are counted once when both interiors lie
    // on the same side. Their integral is zero in this translated frame.
    double result=arc_integral(a,b,ga,deficit)+arc_integral(b,a,gb,deficit)
        +line_integral(a,b,ga)+line_integral(b,a,gb);
    if(result<=0){if(ga)std::fill(ga,ga+3,0.0);if(gb)std::fill(gb,gb+3,0.0);}
    return std::max(0.0,result);
}
}
extern "C" {
void packing_polygon_areas_jacobian(int n,const double* p,double* out,double* jac) {
    int count=n*(n-1)/2,cols=n*3,k=0;
    if(jac)std::fill(jac,jac+count*cols,0.0);
    for(int i=0;i<n;++i) for(int j=i+1;j<n;++j){
        double deficit=0;
        double result=area(p+3*i,p+3*j,jac?jac+k*cols+3*i:nullptr,jac?jac+k*cols+3*j:nullptr,&deficit);
        out[k]=std::max(0.0,result-deficit);
        if(jac && result>0 && dot(V{p[3*j]-p[3*i],p[3*j+1]-p[3*i+1]},V{p[3*j]-p[3*i],p[3*j+1]-p[3*i+1]})>=1e-20){
            V origin{p[3*i],p[3*i+1]};S a=shape(p+3*i,origin),b=shape(p+3*j,origin);
            arc_deficit_gradient(a,b,jac+k*cols+3*i,jac+k*cols+3*j);
            arc_deficit_gradient(b,a,jac+k*cols+3*j,jac+k*cols+3*i);
        }
        if(jac && out[k]<=0){
            std::fill(jac+k*cols+3*i,jac+k*cols+3*i+3,0.0);
            std::fill(jac+k*cols+3*j,jac+k*cols+3*j+3,0.0);
        }
        ++k;
    }
}
void packing_polygon_areas(int n,const double* p,double* out) {
    packing_polygon_areas_jacobian(n,p,out,nullptr);
}
void packing_areas_jacobian(int n,const double* p,double* out,double* jac) {
    int count=n*(n-1)/2,cols=n*3,k=0;
    std::fill(jac,jac+count*cols,0.0);
    for(int i=0;i<n;++i) for(int j=i+1;j<n;++j){
        out[k]=area(p+3*i,p+3*j,jac+k*cols+3*i,jac+k*cols+3*j);
        ++k;
    }
}
void packing_areas(int n,const double* p,double* out) {
    int k=0;
    for(int i=0;i<n;++i) for(int j=i+1;j<n;++j)
        out[k++]=area(p+3*i,p+3*j);
}
}
