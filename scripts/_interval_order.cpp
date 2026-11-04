#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

// All-zero node closes a tour. Its Hamming length is exactly twice the
// number of true runs in the ordered column masks.
extern "C" void interval_hamming_two_opt(int n,int words,const std::uint64_t* bits,
 const std::int64_t* initial,int maximum_iterations,double seconds,
 std::int64_t* output,std::int64_t* stats){
 const int size=n+1;std::vector<std::uint32_t>d(size*size,0);
 for(int a=0;a<size;++a)for(int b=a+1;b<size;++b){
  std::uint32_t value=0;
  for(int k=0;k<words;++k)value+=__builtin_popcountll((a==n?0:bits[a*words+k])^(b==n?0:bits[b*words+k]));
  d[a*size+b]=d[b*size+a]=value;
 }
 std::vector<int>tour(size);tour[0]=n;for(int k=0;k<n;++k)tour[k+1]=initial[k];
 auto cost=[&](){std::int64_t v=0;for(int k=0;k<size;++k)v+=d[tour[k]*size+tour[(k+1)%size]];return v;};
 stats[0]=cost();stats[2]=0;auto start=std::chrono::steady_clock::now();
 for(int iteration=0;iteration<maximum_iterations;++iteration){
  if(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>=seconds)break;
  std::int64_t best=0;int first=-1,last=-1;
  for(int i=0;i<size-2;++i)for(int j=i+2;j<size;++j){
   const int a=tour[i],b=tour[i+1],c=tour[j],e=tour[(j+1)%size];
   const std::int64_t change=static_cast<std::int64_t>(d[a*size+c])+d[b*size+e]-d[a*size+b]-d[c*size+e];
   if(change<best){best=change;first=i+1;last=j;}
  }
  if(first<0)break;std::reverse(tour.begin()+first,tour.begin()+last+1);++stats[2];
 }
 stats[1]=cost();for(int k=0;k<n;++k)output[k]=tour[k+1];
}
