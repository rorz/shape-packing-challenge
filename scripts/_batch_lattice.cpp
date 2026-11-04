#include <cmath>
#include <cstdint>
#include <vector>
#include <unordered_set>
#include <algorithm>

struct Key { std::uint64_t a,b; bool operator==(const Key& other)const{return a==other.a&&b==other.b;} };
struct Hasher { std::size_t operator()(const Key& key)const{return key.a^(key.b+0x9e3779b97f4a7c15ULL+(key.a<<6)+(key.a>>2));} };
static std::unordered_set<Key,Hasher> seen;
static std::unordered_set<Key,Hasher> seen_centers;
static std::uint64_t mix(std::uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}

extern "C" {
void packing_batch_babai(int count,int dimensions,const double* targets,const double* mu,
    const std::int64_t* moves,const std::int64_t* anchor,std::int64_t* coefficients,
    std::int64_t* units,double* placements,int* ambiguous) {
    std::vector<double> residual(dimensions);
    std::vector<std::int64_t> delta(45);
    for(int row=0;row<count;++row){
        std::copy(targets+row*dimensions,targets+(row+1)*dimensions,residual.begin());
        std::int64_t* c=coefficients+row*dimensions;ambiguous[row]=0;
        for(int j=dimensions-1;j>=0;--j){
            double value=residual[j],fraction=value-std::floor(value);
            if(std::abs(fraction-0.5)<2e-7)ambiguous[row]=1;
            c[j]=std::llrint(value); // nearest-even, matching fpylll Babai ties
            for(int k=0;k<j;++k)residual[k]-=static_cast<double>(c[j])*mu[j*dimensions+k];
        }
        std::fill(delta.begin(),delta.end(),0);
        for(int j=0;j<dimensions;++j)for(int k=0;k<45;++k)delta[k]+=c[j]*moves[j*45+k];
        const std::int64_t dx=delta[0],dy=delta[1];
        std::int64_t* out=units+row*45;double* p=placements+row*45;
        for(int piece=0;piece<15;++piece){
            out[3*piece]=anchor[3*piece]+delta[3*piece]-dx;
            out[3*piece+1]=anchor[3*piece+1]+delta[3*piece+1]-dy;
            double theta=(anchor[3*piece+2]+delta[3*piece+2])*1e-6;
            out[3*piece+2]=std::llrint(std::atan2(std::sin(theta),std::cos(theta))*1e6);
            for(int k=0;k<3;++k)p[3*piece+k]=out[3*piece+k]*1e-6;
        }
    }
}
// Heuristic stochastic nearest plane, deliberately separate from the unchanged
// deterministic API. Choices are ranked using a deterministic first pass;
// their actual alternative is recomputed in the second pass after propagation.
void packing_batch_stochastic(int count,int dimensions,const double* targets,const double* mu,
    const double* norms,const std::int64_t* moves,const std::int64_t* anchor,
    std::uint64_t seed,int max_flips,double temperature,double norm_power,double fraction,
    std::int64_t* coefficients,std::int64_t* units,double* placements,int* ambiguous,
    int* changes,double* distances) {
    if(max_flips<=0||temperature<=0||fraction<=0){
        packing_batch_babai(count,dimensions,targets,mu,moves,anchor,coefficients,units,placements,ambiguous);
        std::fill(changes,changes+count,0);std::fill(distances,distances+2*count,0.0);return;
    }
    max_flips=std::min({max_flips,dimensions,3});
    std::vector<double> residual(dimensions),margin(dimensions),lognorm(dimensions);
    std::vector<int> selected(dimensions),eligible(dimensions);
    std::vector<std::pair<double,int>> ranked;ranked.reserve(dimensions);
    std::vector<std::int64_t> delta(45);
    for(int j=0;j<dimensions;++j){
        lognorm[j]=std::log(std::max(norms[j],1e-100));eligible[j]=0;
        for(int k=0;k<45;++k){
            std::int64_t v=moves[j*45+k];if(k%3<2)v-=moves[j*45+k%3];
            if(v!=0){eligible[j]=1;break;}
        }
    }
    for(int row=0;row<count;++row){
        std::uint64_t state=mix(seed^mix(static_cast<std::uint64_t>(row)+0x9e3779b97f4a7c15ULL));
        auto uniform=[&](){state+=0x9e3779b97f4a7c15ULL;return std::min(std::nextafter(1.0,0.0),((mix(state)>>11)+0.5)*0x1.0p-53);};
        std::copy(targets+row*dimensions,targets+(row+1)*dimensions,residual.begin());
        double original_distance=0.0;ambiguous[row]=0;
        for(int j=dimensions-1;j>=0;--j){
            const double value=residual[j],nearest=std::nearbyint(value),d=value-nearest;
            margin[j]=std::max(0.0,1-2*std::abs(d));original_distance+=d*d*norms[j];
            if(std::abs(margin[j])<4e-7)ambiguous[row]=1;
            for(int k=0;k<j;++k)residual[k]-=nearest*mu[j*dimensions+k];
        }
        ranked.clear();std::fill(selected.begin(),selected.end(),0);
        if(uniform()<fraction){
            const int wanted=1+static_cast<int>(uniform()*max_flips);
            for(int j=0;j<dimensions;++j)if(eligible[j]){
                // Weighted sampling without replacement via exponential races.
                // This is a proposal heuristic, not an exact lattice sampler.
                const double score=std::log(-std::log(uniform()))+margin[j]/temperature-norm_power*lognorm[j];
                ranked.emplace_back(score,j);
            }
            const int used=std::min(wanted,static_cast<int>(ranked.size()));
            std::partial_sort(ranked.begin(),ranked.begin()+used,ranked.end());
            for(int k=0;k<used;++k)selected[ranked[k].second]=1;
        }
        std::copy(targets+row*dimensions,targets+(row+1)*dimensions,residual.begin());
        std::int64_t* c=coefficients+row*dimensions;changes[row]=0;double distance=0.0;
        for(int j=dimensions-1;j>=0;--j){
            const double value=residual[j];c[j]=std::llrint(value);
            if(selected[j]){c[j]+=value>c[j]?1:(value<c[j]?-1:(uniform()<0.5?-1:1));++changes[row];}
            const double d=value-c[j];distance+=d*d*norms[j];
            for(int k=0;k<j;++k)residual[k]-=static_cast<double>(c[j])*mu[j*dimensions+k];
        }
        distances[2*row]=original_distance;distances[2*row+1]=distance;
        std::fill(delta.begin(),delta.end(),0);
        for(int j=0;j<dimensions;++j)for(int k=0;k<45;++k)delta[k]+=c[j]*moves[j*45+k];
        const std::int64_t dx=delta[0],dy=delta[1];
        std::int64_t* out=units+row*45;double* p=placements+row*45;
        for(int piece=0;piece<15;++piece){
            out[3*piece]=anchor[3*piece]+delta[3*piece]-dx;
            out[3*piece+1]=anchor[3*piece+1]+delta[3*piece+1]-dy;
            const double theta=(anchor[3*piece+2]+delta[3*piece+2])*1e-6;
            out[3*piece+2]=std::llrint(std::atan2(std::sin(theta),std::cos(theta))*1e6);
            for(int k=0;k<3;++k)p[3*piece+k]=out[3*piece+k]*1e-6;
        }
    }
}
void packing_batch_unique(int count,const std::int64_t* units,std::int64_t limit,int* keep,std::int64_t* info){
    std::int64_t duplicates=0,resets=0;
    for(int row=0;row<count;++row){
        if(static_cast<std::int64_t>(seen.size())>=limit){seen.clear();++resets;}
        const std::int64_t* u=units+row*45;
        std::uint64_t a=0x243f6a8885a308d3ULL,b=0x13198a2e03707344ULL;
        for(int k=0;k<45;++k){
            std::int64_t value=u[k];if(k%3<2)value-=u[k%3];
            a=mix(a^static_cast<std::uint64_t>(value));
            b=mix(b+static_cast<std::uint64_t>(value)+0x9e3779b97f4a7c15ULL+k);
        }
        keep[row]=seen.insert({a,b}).second?1:0;duplicates+=1-keep[row];
    }
    info[0]=duplicates;info[1]=seen.size();info[2]=resets;
}
void packing_batch_unique_centers(int count,const std::int64_t* units,std::int64_t limit,int* keep,std::int64_t* info){
    std::int64_t duplicates=0,resets=0;
    for(int row=0;row<count;++row){
        if(static_cast<std::int64_t>(seen_centers.size())>=limit){seen_centers.clear();++resets;}
        const std::int64_t* u=units+row*45;
        std::uint64_t a=0x243f6a8885a308d3ULL,b=0x13198a2e03707344ULL;
        for(int k=0;k<45;++k)if(k%3<2){
            const std::int64_t value=u[k]-u[k%3];
            a=mix(a^static_cast<std::uint64_t>(value));b=mix(b+static_cast<std::uint64_t>(value)+0x9e3779b97f4a7c15ULL+k);
        }
        keep[row]=seen_centers.insert({a,b}).second?1:0;duplicates+=1-keep[row];
    }
    info[0]=duplicates;info[1]=seen_centers.size();info[2]=resets;
}
void packing_batch_reset(){seen.clear();seen_centers.clear();}
}
