#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// Pair tables use i<j lexicographic order, then (state_i,state_j).
extern "C" void packing_min_binary_cost(int n, const double* unary,
    const double* pairs, const double* preference, int* assignment, double* output) {
    std::vector<double> table(n*n*4, 0.0);
    int row=0;
    double largest=0.0;
    for(int i=0;i<n;++i) {
        for(int a=0;a<2;++a)largest=std::max(largest,unary[2*i+a]);
        for(int j=i+1;j<n;++j,++row)for(int a=0;a<2;++a)for(int b=0;b<2;++b) {
            double value=pairs[4*row+2*a+b];
            table[4*(i*n+j)+2*a+b]=value;
            table[4*(j*n+i)+2*b+a]=value;
            largest=std::max(largest,value);
        }
    }
    std::vector<int> bits(n),best(n);
    double minimum_preference=0;
    for(int i=0;i<n;++i) {
        bits[i]=preference[2*i+1]<preference[2*i]?1:0;
        minimum_preference+=preference[2*i+bits[i]];
    }
    auto exact_energy=[&]() {
        double value=0;
        for(int i=0;i<n;++i) {
            value+=unary[2*i+bits[i]];
            for(int j=i+1;j<n;++j)value+=table[4*(i*n+j)+2*bits[i]+bits[j]];
        }
        return value;
    };
    auto exact_preference=[&]() {
        double value=0;for(int i=0;i<n;++i)value+=preference[2*i+bits[i]];return value;
    };
    double energy=exact_energy(),best_energy=energy,best_preference=exact_preference();
    best=bits;
    std::uint64_t visited=1,total=std::uint64_t(1)<<n;
    // Exact nonnegative rescoring near an incumbent and every 256 flips prevents
    // accumulated subtraction error from fabricating a feasible assignment.
    const double guard=largest*std::numeric_limits<double>::epsilon()*1024*(n+1)*(n+1)*(n+1);
    if(!(best_energy==0 && best_preference==minimum_preference)) {
        for(std::uint64_t step=1;step<total;++step) {
            int changed=__builtin_ctzll(step),old=bits[changed],fresh=1-old;
            energy+=unary[2*changed+fresh]-unary[2*changed+old];
            for(int j=0;j<n;++j)if(j!=changed)
                energy+=table[4*(changed*n+j)+2*fresh+bits[j]]
                       -table[4*(changed*n+j)+2*old+bits[j]];
            bits[changed]=fresh;++visited;
            if((step&255)==0)energy=exact_energy();
            if(energy<=best_energy+guard) {
                double actual=exact_energy();
                if(actual<best_energy) {
                    best_energy=actual;best_preference=exact_preference();best=bits;
                } else if(actual==best_energy) {
                    double pref=exact_preference();
                    if(pref<best_preference){best_preference=pref;best=bits;}
                }
                if(best_energy==0 && best_preference==minimum_preference)break;
            }
        }
    }
    std::copy(best.begin(),best.end(),assignment);
    output[0]=best_energy;output[1]=best_preference;output[2]=double(visited);
}
