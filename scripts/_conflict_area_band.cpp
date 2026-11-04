#include "../src/semicircle_packing/area_kernel.cpp"

namespace {
double local_lost_area(const double* ap,const double* bp) {
    constexpr double half_step=PI/(2*4095);
    const double sagitta=1-std::cos(half_step);
    const double annulus_factor=.5*std::pow(std::sin(half_step),2);
    const double full_deficit=.5*(PI-4095*std::sin(PI/4095));
    // A lost polygon point is at most sagitta from its outer circular point.
    // Such an outer point must belong to the other disk and halfplane after
    // expansion by sagitta. Extra numeric expansion can only enlarge this set.
    const double expansion=sagitta+1e-12;
    const double start=ap[2]-PI/2;
    const double dx=bp[0]-ap[0],dy=bp[1]-ap[1];
    const double distance=std::hypot(dx,dy);
    double circle_limit=-2;
    std::vector<double> cuts{start,start+PI};
    if(distance>expansion) {
        circle_limit=(distance*distance-2*expansion-expansion*expansion)/(2*distance);
        if(circle_limit>=1)return 1e-14;
        if(circle_limit>-1) {
            double angle=std::atan2(dy,dx),off=std::acos(circle_limit);
            arc_cut(cuts,angle-off,start);arc_cut(cuts,angle+off,start);
        }
    }
    double half_limit=dx*std::cos(bp[2])+dy*std::sin(bp[2])-expansion;
    if(half_limit>=1)return 1e-14;
    if(half_limit>-1) {
        double off=std::acos(half_limit);
        arc_cut(cuts,bp[2]-off,start);arc_cut(cuts,bp[2]+off,start);
    }
    std::sort(cuts.begin(),cuts.end());
    double measure=0;
    double disk_angle=std::atan2(dy,dx);
    for(size_t i=1;i<cuts.size();++i) {
        double mid=.5*(cuts[i-1]+cuts[i]);
        if((circle_limit<=-1 || std::cos(mid-disk_angle)>=circle_limit-1e-13)
           &&(half_limit<=-1 || std::cos(mid-bp[2])>=half_limit-1e-13))
            measure+=cuts[i]-cuts[i-1];
    }
    return std::min(full_deficit,annulus_factor*std::min(PI,measure+1e-8))+1e-14;
}
}

extern "C" void packing_true_area_loss_band(int n,int m,const double* a,const double* b,double* out,double* loss) {
    for(int i=0;i<n;++i)for(int j=0;j<m;++j) {
        const double* aa=a+3*i;const double* bb=b+3*j;
        out[i*m+j]=area(aa,bb);
        loss[i*m+j]=local_lost_area(aa,bb)+local_lost_area(bb,aa);
    }
}
