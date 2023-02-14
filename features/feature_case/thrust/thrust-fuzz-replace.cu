#include <thrust/device_vector.h>                                                                 
#include <thrust/host_vector.h>                                                                   
#include <thrust/logical.h>                                                                       
#include <thrust/functional.h>                                                                    
#include <thrust/adjacent_difference.h>                                                           
#include <thrust/sort.h>                                                                          
#include <thrust/unique.h>                                                                        
                                                                                                  
struct s_pred_A                                                                    
{                                                                                               
  __host__ __device__                                                                           
  bool operator()(long long x) const                                              
  {                                                                                             
    return x >1018;                                
  }                                                                                             
  typedef long long argument_type;                                                
};                                                                                              
static s_pred_A pred_A;                                               
struct s_pred_B                                                                    
{                                                                                               
  __host__ __device__                                                                           
  bool operator()(long long x) const                                              
  {                                                                                             
    return x <=1562;                                
  }                                                                                             
  typedef long long argument_type;                                                
};                                                                                              
static s_pred_B pred_B;                                               
struct s_pred_C                                                                    
{                                                                                               
  __host__ __device__                                                                           
  bool operator()(long long x) const                                              
  {                                                                                             
    return x !=761;                                
  }                                                                                             
  typedef long long argument_type;                                                
};                                                                                              
static s_pred_C pred_C;                                               
struct s_pred_D                                                                    
{                                                                                               
  __host__ __device__                                                                           
  bool operator()(long long x) const                                              
  {                                                                                             
    return x <1052;                                
  }                                                                                             
  typedef long long argument_type;                                                
};                                                                                              
static s_pred_D pred_D;                                               
struct s_pred_E                                                                    
{                                                                                               
  __host__ __device__                                                                           
  bool operator()(long long x) const                                              
  {                                                                                             
    return x >=1726;                                
  }                                                                                             
  typedef long long argument_type;                                                
};                                                                                              
static s_pred_E pred_E;                                               
                                                                                                  
#ifdef GENERATE_REFERENCE                                                                         
  #define checkVector(VEC,RESULT_NAME,FILE,LINE)                                                \
    checkVectorInternal(VEC,#RESULT_NAME,FILE,LINE)                                               
#else                                                                                             
  #include "thrust-fuzz-replace.h"                                                                        
                                                                                                  
  #define checkVector(VEC,RESULT_NAME,FILE,LINE)                                                \
    checkVectorInternal(VEC,#RESULT_NAME,RESULT_NAME,FILE,LINE)                                   
#endif                                                                                            
                                                                                                  
template<typename T>                                                                              
void printVector(T &V, int index) {                                                               
  for (auto i=0; i<17; i++) {                                
    std::cerr << "V" << index << "[" << i << "] = 0x" << std::hex << V[i] << ";\n";      
  }                                                                                               
}                                                                                                 
                                                                                                  
template<typename T>                                                                              
void checkVectorInternal(T &V,                                                                    
                         const char  *result_name,                                                
#ifndef GENERATE_REFERENCE                                                                        
                         unsigned long long check_value,                                          
#endif                                                                                            
                         const char  *file,                                                       
                         int          line) {                                                     
  unsigned long long xor_acc{0};                                                                  
                                                                                                  
  for (auto i=0; i<17; i++) {                                                
    xor_acc^=(static_cast<unsigned long long>(V[i])*(2*i+1));                                     
  }                                                                                               
#ifdef GENERATE_REFERENCE                                                                         
  std::cout << "#define " << result_name << " " << xor_acc << "ULL\n";                     
#else                                                                                             
  std::string msg{file};                                                                          
                                                                                                  
  if (xor_acc!=check_value) {                                                                     
    std::cerr << msg << ":" <<  std::to_string(line) << " " << result_name << "\n";        
    exit(-1);                                                                                     
  }                                                                                               
#endif                                                                                            
}                                                                                                 
// seed 1

int main() {
  thrust::host_vector<long long> V0(17);
  thrust::host_vector<long long> V1(17);
  thrust::host_vector<long long> V2(17);
  thrust::host_vector<long long> V3(17);
  thrust::host_vector<long long> V4(17);
  thrust::host_vector<long long> V5(17);
  thrust::host_vector<long long> V6(17);
  thrust::host_vector<long long> V7(17);
  thrust::host_vector<long long> V8(17);
  thrust::host_vector<long long> V9(17);

  V0[0]=328; V0[1]=-702; V0[2]=791; V0[3]=677; V0[4]=671; V0[5]=721; V0[6]=186; V0[7]=-13; V0[8]=80; V0[9]=-416; V0[10]=309; V0[11]=-115; V0[12]=845; V0[13]=-126; V0[14]=886; V0[15]=-956; V0[16]=183; 
  V1[0]=-80; V1[1]=259; V1[2]=-220; V1[3]=-646; V1[4]=360; V1[5]=818; V1[6]=976; V1[7]=413; V1[8]=-815; V1[9]=568; V1[10]=-1; V1[11]=163; V1[12]=499; V1[13]=-824; V1[14]=887; V1[15]=257; V1[16]=-10; 
  V2[0]=-919; V2[1]=631; V2[2]=-24; V2[3]=-829; V2[4]=176; V2[5]=549; V2[6]=-349; V2[7]=194; V2[8]=-354; V2[9]=586; V2[10]=-566; V2[11]=-517; V2[12]=-220; V2[13]=354; V2[14]=-963; V2[15]=590; V2[16]=-287; 
  V3[0]=-811; V3[1]=608; V3[2]=90; V3[3]=-398; V3[4]=-480; V3[5]=-816; V3[6]=-897; V3[7]=-725; V3[8]=508; V3[9]=-272; V3[10]=-477; V3[11]=371; V3[12]=-689; V3[13]=314; V3[14]=172; V3[15]=-54; V3[16]=530; 
  V4[0]=-163; V4[1]=142; V4[2]=-148; V4[3]=828; V4[4]=503; V4[5]=-198; V4[6]=-99; V4[7]=-219; V4[8]=260; V4[9]=-323; V4[10]=-60; V4[11]=759; V4[12]=-718; V4[13]=-73; V4[14]=-661; V4[15]=-997; V4[16]=202; 
  V5[0]=358; V5[1]=753; V5[2]=-116; V5[3]=-416; V5[4]=344; V5[5]=486; V5[6]=576; V5[7]=821; V5[8]=-762; V5[9]=424; V5[10]=530; V5[11]=837; V5[12]=846; V5[13]=-24; V5[14]=-947; V5[15]=236; V5[16]=480; 
  V6[0]=-746; V6[1]=119; V6[2]=-978; V6[3]=310; V6[4]=827; V6[5]=381; V6[6]=-168; V6[7]=-815; V6[8]=-740; V6[9]=502; V6[10]=-479; V6[11]=405; V6[12]=154; V6[13]=252; V6[14]=-544; V6[15]=296; V6[16]=76; 
  V7[0]=533; V7[1]=485; V7[2]=-285; V7[3]=-365; V7[4]=-892; V7[5]=902; V7[6]=722; V7[7]=470; V7[8]=-400; V7[9]=-156; V7[10]=-725; V7[11]=-314; V7[12]=-604; V7[13]=490; V7[14]=-805; V7[15]=964; V7[16]=419; 
  V8[0]=-225; V8[1]=938; V8[2]=-752; V8[3]=-802; V8[4]=-133; V8[5]=-456; V8[6]=482; V8[7]=876; V8[8]=758; V8[9]=324; V8[10]=159; V8[11]=-162; V8[12]=-322; V8[13]=73; V8[14]=-865; V8[15]=379; V8[16]=126; 
  V9[0]=-925; V9[1]=-191; V9[2]=-900; V9[3]=-199; V9[4]=57; V9[5]=553; V9[6]=-197; V9[7]=-389; V9[8]=515; V9[9]=762; V9[10]=521; V9[11]=17; V9[12]=-629; V9[13]=887; V9[14]=897; V9[15]=724; V9[16]=326; 


  thrust::replace_if(thrust::host,V8.begin(),V8.end(),thrust::not1(pred_D),-803);
  checkVector(V8,CHECK1,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V0.begin()+1,V0.end()-1,V9.begin(),pred_B,-349);
  checkVector(V9,CHECK2,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V9.begin()+1,V9.end()-1,V6.begin(),V2.begin(),pred_E,712);
  checkVector(V2,CHECK3,__FILE__,__LINE__);
  thrust::replace_copy_if(V5.begin(),V5.end()-1,V6.begin(),V7.begin(),pred_A,82);
  checkVector(V7,CHECK4,__FILE__,__LINE__);
  thrust::replace_if(V1.begin(),V1.end(),thrust::not1(pred_C),212);
  checkVector(V1,CHECK5,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V8.begin(),V8.end()-1,V1.begin(),V2.begin(),pred_E,-863);
  checkVector(V2,CHECK6,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V0.begin()+1,V0.end(),V2.begin(),thrust::not1(pred_A),27);
  checkVector(V0,CHECK7,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V1.begin()+1,V1.end(),V9.begin(),thrust::not1(pred_E),962);
  checkVector(V1,CHECK8,__FILE__,__LINE__);
  thrust::replace_copy_if(V8.begin(),V8.end(),V2.begin(),V3.begin(),thrust::not1(pred_D),943);
  checkVector(V3,CHECK9,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V1.begin()+1,V1.end(),V4.begin(),V9.begin(),pred_C,667);
  checkVector(V9,CHECK10,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V7.begin(),V7.begin()+5,pred_C,773);
  checkVector(V7,CHECK11,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V8.begin()+1,V8.end(),V5.begin(),pred_C,-173);
  checkVector(V5,CHECK12,__FILE__,__LINE__);
  thrust::replace_copy_if(V1.begin(),V1.end()-1,V8.begin(),V6.begin(),pred_E,16);
  checkVector(V6,CHECK13,__FILE__,__LINE__);
  thrust::replace_copy_if(V9.begin()+1,V9.end()-1,V5.begin(),pred_A,702);
  checkVector(V5,CHECK14,__FILE__,__LINE__);
  thrust::replace_if(V5.begin(),V5.end(),pred_E,764);
  checkVector(V5,CHECK15,__FILE__,__LINE__);
  thrust::replace_copy_if(V0.begin(),V0.begin()+5,V1.begin(),V2.begin(),pred_D,-511);
  checkVector(V2,CHECK16,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V2.begin()+1,V2.end()-1,pred_B,660);
  checkVector(V2,CHECK17,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V5.begin()+1,V5.end(),V9.begin(),pred_B,756);
  checkVector(V9,CHECK18,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V1.begin(),V1.end()-1,V9.begin(),V6.begin(),thrust::not1(pred_A),-62);
  checkVector(V6,CHECK19,__FILE__,__LINE__);
  thrust::replace_copy_if(V8.begin(),V8.end(),V3.begin(),pred_E,-615);
  checkVector(V3,CHECK20,__FILE__,__LINE__);
  thrust::replace_copy_if(V4.begin(),V4.end()-1,V3.begin(),V7.begin(),thrust::not1(pred_C),893);
  checkVector(V7,CHECK21,__FILE__,__LINE__);
  thrust::replace_if(V9.begin(),V9.end()-1,V4.begin(),pred_D,-595);
  checkVector(V9,CHECK22,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V8.begin()+1,V8.end()-1,pred_A,211);
  checkVector(V8,CHECK23,__FILE__,__LINE__);
  thrust::replace_copy_if(V0.begin(),V0.end()-1,V9.begin(),pred_E,-644);
  checkVector(V9,CHECK24,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V2.begin(),V2.end(),V6.begin(),pred_E,-217);
  checkVector(V6,CHECK25,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V6.begin()+1,V6.end(),V0.begin(),pred_E,-572);
  checkVector(V0,CHECK26,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V0.begin(),V0.begin()+5,V9.begin(),pred_D,-40);
  checkVector(V9,CHECK27,__FILE__,__LINE__);
  thrust::replace_if(V6.begin()+1,V6.end(),pred_E,-470);
  checkVector(V6,CHECK28,__FILE__,__LINE__);
  thrust::replace_copy_if(V3.begin(),V3.begin()+5,V5.begin(),pred_A,594);
  checkVector(V5,CHECK29,__FILE__,__LINE__);
  thrust::replace_if(V6.begin(),V6.end()-1,pred_C,85);
  checkVector(V6,CHECK30,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V6.begin()+1,V6.end(),V5.begin(),V9.begin(),thrust::not1(pred_D),-188);
  checkVector(V9,CHECK31,__FILE__,__LINE__);
  thrust::replace_if(V4.begin()+1,V4.end(),V0.begin(),pred_E,-214);
  checkVector(V4,CHECK32,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V9.begin(),V9.end(),V0.begin(),pred_A,-517);
  checkVector(V0,CHECK33,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V6.begin(),V6.end(),thrust::not1(pred_D),-835);
  checkVector(V6,CHECK34,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V6.begin(),V6.end()-1,V7.begin(),thrust::not1(pred_C),-200);
  checkVector(V6,CHECK35,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V3.begin(),V3.end()-1,pred_E,761);
  checkVector(V3,CHECK36,__FILE__,__LINE__);
  thrust::replace_if(V0.begin()+1,V0.end(),V7.begin(),thrust::not1(pred_D),-288);
  checkVector(V0,CHECK37,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V0.begin(),V0.end()-1,V6.begin(),V4.begin(),pred_D,-867);
  checkVector(V4,CHECK38,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V4.begin()+1,V4.end(),pred_E,-446);
  checkVector(V4,CHECK39,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V1.begin(),V1.end(),V2.begin(),pred_A,245);
  checkVector(V2,CHECK40,__FILE__,__LINE__);
  thrust::replace_copy_if(V3.begin()+1,V3.end()-1,V4.begin(),pred_C,413);
  checkVector(V4,CHECK41,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V9.begin()+1,V9.end()-1,thrust::not1(pred_E),-450);
  checkVector(V9,CHECK42,__FILE__,__LINE__);
  thrust::replace_if(V5.begin(),V5.begin()+5,pred_A,-624);
  checkVector(V5,CHECK43,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V2.begin()+1,V2.end(),pred_E,-516);
  checkVector(V2,CHECK44,__FILE__,__LINE__);
  thrust::replace_if(V4.begin(),V4.end(),pred_D,49);
  checkVector(V4,CHECK45,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V7.begin(),V7.end(),V0.begin(),thrust::not1(pred_E),908);
  checkVector(V7,CHECK46,__FILE__,__LINE__);
  thrust::replace_copy_if(V6.begin(),V6.end()-1,V1.begin(),V8.begin(),pred_D,808);
  checkVector(V8,CHECK47,__FILE__,__LINE__);
  thrust::replace_copy_if(V3.begin()+1,V3.end()-1,V8.begin(),V6.begin(),pred_E,-435);
  checkVector(V6,CHECK48,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V5.begin(),V5.begin()+5,V4.begin(),pred_B,463);
  checkVector(V4,CHECK49,__FILE__,__LINE__);
  thrust::replace_if(V9.begin(),V9.end(),pred_B,271);
  checkVector(V9,CHECK50,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V9.begin(),V9.end(),V0.begin(),pred_A,-545);
  checkVector(V0,CHECK51,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V3.begin(),V3.end()-1,thrust::not1(pred_C),811);
  checkVector(V3,CHECK52,__FILE__,__LINE__);
  thrust::replace_copy_if(V6.begin()+1,V6.end()-1,V1.begin(),V2.begin(),pred_A,-470);
  checkVector(V2,CHECK53,__FILE__,__LINE__);
  thrust::replace_copy_if(V3.begin(),V3.end(),V9.begin(),pred_C,-721);
  checkVector(V9,CHECK54,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V5.begin(),V5.end()-1,V0.begin(),pred_E,-550);
  checkVector(V0,CHECK55,__FILE__,__LINE__);
  thrust::replace_if(V2.begin()+1,V2.end(),V4.begin(),pred_C,43);
  checkVector(V2,CHECK56,__FILE__,__LINE__);
  thrust::replace_copy_if(V4.begin(),V4.end()-1,V6.begin(),pred_E,82);
  checkVector(V6,CHECK57,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V3.begin()+1,V3.end(),V3.begin(),V5.begin(),thrust::not1(pred_E),-491);
  checkVector(V5,CHECK58,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V4.begin()+1,V4.end()-1,pred_D,955);
  checkVector(V4,CHECK59,__FILE__,__LINE__);
  thrust::replace_if(V8.begin()+1,V8.end(),pred_B,504);
  checkVector(V8,CHECK60,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V2.begin(),V2.begin()+5,V6.begin(),V3.begin(),pred_B,-787);
  checkVector(V3,CHECK61,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V7.begin(),V7.begin()+5,V1.begin(),pred_C,-293);
  checkVector(V7,CHECK62,__FILE__,__LINE__);
  thrust::replace_copy_if(V3.begin(),V3.end(),V8.begin(),pred_E,-548);
  checkVector(V8,CHECK63,__FILE__,__LINE__);
  thrust::replace_if(V4.begin(),V4.end()-1,V2.begin(),thrust::not1(pred_E),-881);
  checkVector(V4,CHECK64,__FILE__,__LINE__);
  thrust::replace_if(V9.begin(),V9.begin()+5,V2.begin(),pred_E,-456);
  checkVector(V9,CHECK65,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V0.begin(),V0.end()-1,pred_C,754);
  checkVector(V0,CHECK66,__FILE__,__LINE__);
  thrust::replace_copy_if(V8.begin(),V8.end()-1,V1.begin(),pred_C,331);
  checkVector(V1,CHECK67,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V8.begin(),V8.begin()+5,thrust::not1(pred_E),-315);
  checkVector(V8,CHECK68,__FILE__,__LINE__);
  thrust::replace_if(V7.begin(),V7.end()-1,thrust::not1(pred_C),-284);
  checkVector(V7,CHECK69,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V3.begin()+1,V3.end(),V6.begin(),V8.begin(),thrust::not1(pred_D),-918);
  checkVector(V8,CHECK70,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V7.begin(),V7.end()-1,V4.begin(),pred_D,-6);
  checkVector(V7,CHECK71,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V0.begin(),V0.begin()+5,pred_C,743);
  checkVector(V0,CHECK72,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V0.begin(),V0.end()-1,V8.begin(),pred_D,590);
  checkVector(V8,CHECK73,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V0.begin(),V0.end()-1,pred_C,-235);
  checkVector(V0,CHECK74,__FILE__,__LINE__);
  thrust::replace_copy_if(V9.begin()+1,V9.end()-1,V2.begin(),V5.begin(),thrust::not1(pred_C),-843);
  checkVector(V5,CHECK75,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V7.begin()+1,V7.end(),V6.begin(),thrust::not1(pred_A),949);
  checkVector(V7,CHECK76,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V0.begin(),V0.begin()+5,V5.begin(),pred_C,-200);
  checkVector(V0,CHECK77,__FILE__,__LINE__);
  thrust::replace_copy_if(V8.begin(),V8.begin()+5,V7.begin(),V0.begin(),thrust::not1(pred_D),-375);
  checkVector(V0,CHECK78,__FILE__,__LINE__);
  thrust::replace_copy_if(V2.begin(),V2.begin()+5,V8.begin(),pred_E,-414);
  checkVector(V8,CHECK79,__FILE__,__LINE__);
  thrust::replace_copy_if(V0.begin(),V0.end(),V1.begin(),V4.begin(),pred_A,-564);
  checkVector(V4,CHECK80,__FILE__,__LINE__);
  thrust::replace_copy_if(V5.begin(),V5.begin()+5,V4.begin(),V6.begin(),pred_A,451);
  checkVector(V6,CHECK81,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V9.begin()+1,V9.end()-1,V4.begin(),V5.begin(),thrust::not1(pred_A),283);
  checkVector(V5,CHECK82,__FILE__,__LINE__);
  thrust::replace_copy_if(V8.begin(),V8.end(),V3.begin(),pred_A,797);
  checkVector(V3,CHECK83,__FILE__,__LINE__);
  thrust::replace_copy_if(V7.begin()+1,V7.end()-1,V7.begin(),V0.begin(),pred_A,12);
  checkVector(V0,CHECK84,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V0.begin()+1,V0.end()-1,V3.begin(),pred_A,412);
  checkVector(V3,CHECK85,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V5.begin(),V5.end()-1,thrust::not1(pred_B),-980);
  checkVector(V5,CHECK86,__FILE__,__LINE__);
  thrust::replace_copy_if(V9.begin(),V9.end(),V7.begin(),V6.begin(),thrust::not1(pred_D),-964);
  checkVector(V6,CHECK87,__FILE__,__LINE__);
  thrust::replace_copy_if(V5.begin()+1,V5.end()-1,V9.begin(),pred_B,-925);
  checkVector(V9,CHECK88,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V6.begin()+1,V6.end()-1,V8.begin(),pred_E,-761);
  checkVector(V8,CHECK89,__FILE__,__LINE__);
  thrust::replace_copy_if(V7.begin(),V7.begin()+5,V4.begin(),V2.begin(),thrust::not1(pred_E),-355);
  checkVector(V2,CHECK90,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V0.begin()+1,V0.end(),pred_A,-799);
  checkVector(V0,CHECK91,__FILE__,__LINE__);
  thrust::replace_copy_if(V4.begin(),V4.end(),V3.begin(),pred_E,-239);
  checkVector(V3,CHECK92,__FILE__,__LINE__);
  thrust::replace_if(V3.begin(),V3.begin()+5,thrust::not1(pred_D),-78);
  checkVector(V3,CHECK93,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V4.begin(),V4.begin()+5,V3.begin(),pred_A,-417);
  checkVector(V3,CHECK94,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V6.begin(),V6.begin()+5,V5.begin(),pred_D,49);
  checkVector(V5,CHECK95,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V8.begin(),V8.end(),V3.begin(),pred_B,641);
  checkVector(V3,CHECK96,__FILE__,__LINE__);
  thrust::replace_copy_if(V4.begin(),V4.end()-1,V6.begin(),pred_E,-312);
  checkVector(V6,CHECK97,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V8.begin(),V8.begin()+5,V2.begin(),V9.begin(),pred_D,825);
  checkVector(V9,CHECK98,__FILE__,__LINE__);
  thrust::replace_copy_if(V0.begin()+1,V0.end(),V6.begin(),V2.begin(),pred_C,536);
  checkVector(V2,CHECK99,__FILE__,__LINE__);
  thrust::replace_copy_if(V4.begin(),V4.end(),V4.begin(),V5.begin(),thrust::not1(pred_E),967);
  checkVector(V5,CHECK100,__FILE__,__LINE__);
  thrust::replace_if(V9.begin()+1,V9.end()-1,V3.begin(),pred_A,395);
  checkVector(V9,CHECK101,__FILE__,__LINE__);
  thrust::replace_copy_if(V1.begin()+1,V1.end(),V8.begin(),V5.begin(),pred_B,-897);
  checkVector(V5,CHECK102,__FILE__,__LINE__);
  thrust::replace_copy_if(V1.begin()+1,V1.end(),V8.begin(),V2.begin(),pred_B,-353);
  checkVector(V2,CHECK103,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V5.begin()+1,V5.end()-1,V5.begin(),V2.begin(),pred_B,735);
  checkVector(V2,CHECK104,__FILE__,__LINE__);
  thrust::replace_copy_if(V6.begin(),V6.begin()+5,V4.begin(),V2.begin(),pred_A,-812);
  checkVector(V2,CHECK105,__FILE__,__LINE__);
  thrust::replace_if(V7.begin(),V7.end(),thrust::not1(pred_E),-348);
  checkVector(V7,CHECK106,__FILE__,__LINE__);
  thrust::replace_copy_if(V3.begin(),V3.end(),V8.begin(),V2.begin(),pred_A,4);
  checkVector(V2,CHECK107,__FILE__,__LINE__);
  thrust::replace_if(V9.begin()+1,V9.end()-1,pred_A,618);
  checkVector(V9,CHECK108,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V7.begin(),V7.begin()+5,pred_A,-326);
  checkVector(V7,CHECK109,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V2.begin(),V2.end(),V7.begin(),pred_D,-38);
  checkVector(V7,CHECK110,__FILE__,__LINE__);
  thrust::replace_copy_if(V3.begin(),V3.begin()+5,V8.begin(),V5.begin(),pred_C,933);
  checkVector(V5,CHECK111,__FILE__,__LINE__);
  thrust::replace_copy_if(V7.begin(),V7.end(),V8.begin(),V0.begin(),pred_B,-603);
  checkVector(V0,CHECK112,__FILE__,__LINE__);
  thrust::replace_if(V8.begin(),V8.end()-1,V6.begin(),thrust::not1(pred_B),-855);
  checkVector(V8,CHECK113,__FILE__,__LINE__);
  thrust::replace_copy_if(V8.begin()+1,V8.end(),V4.begin(),pred_C,468);
  checkVector(V4,CHECK114,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V6.begin()+1,V6.end()-1,V8.begin(),pred_A,455);
  checkVector(V8,CHECK115,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V8.begin(),V8.end()-1,V2.begin(),thrust::not1(pred_B),633);
  checkVector(V8,CHECK116,__FILE__,__LINE__);
  thrust::replace_copy_if(V7.begin()+1,V7.end()-1,V1.begin(),V3.begin(),pred_E,979);
  checkVector(V3,CHECK117,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V2.begin(),V2.end(),V3.begin(),pred_C,211);
  checkVector(V3,CHECK118,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V0.begin()+1,V0.end(),V0.begin(),V8.begin(),pred_A,726);
  checkVector(V8,CHECK119,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V7.begin(),V7.begin()+5,V0.begin(),pred_C,-310);
  checkVector(V0,CHECK120,__FILE__,__LINE__);
  thrust::replace_if(V2.begin(),V2.end()-1,V6.begin(),thrust::not1(pred_B),581);
  checkVector(V2,CHECK121,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V6.begin(),V6.begin()+5,V9.begin(),pred_D,-14);
  checkVector(V6,CHECK122,__FILE__,__LINE__);
  thrust::replace_if(V5.begin(),V5.begin()+5,thrust::not1(pred_C),-830);
  checkVector(V5,CHECK123,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V3.begin()+1,V3.end(),V0.begin(),pred_D,-572);
  checkVector(V0,CHECK124,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V2.begin()+1,V2.end(),V9.begin(),pred_E,-216);
  checkVector(V9,CHECK125,__FILE__,__LINE__);
  thrust::replace_copy_if(V1.begin()+1,V1.end()-1,V9.begin(),pred_B,390);
  checkVector(V9,CHECK126,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V3.begin(),V3.begin()+5,V4.begin(),pred_D,-155);
  checkVector(V3,CHECK127,__FILE__,__LINE__);
  thrust::replace_copy_if(V0.begin(),V0.begin()+5,V6.begin(),pred_D,826);
  checkVector(V6,CHECK128,__FILE__,__LINE__);
  thrust::replace_copy_if(V0.begin()+1,V0.end()-1,V2.begin(),pred_D,382);
  checkVector(V2,CHECK129,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V6.begin(),V6.begin()+5,V9.begin(),thrust::not1(pred_E),-829);
  checkVector(V6,CHECK130,__FILE__,__LINE__);
  thrust::replace_if(V3.begin(),V3.end(),V4.begin(),pred_E,223);
  checkVector(V3,CHECK131,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V9.begin()+1,V9.end(),pred_C,-179);
  checkVector(V9,CHECK132,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V9.begin(),V9.end(),V3.begin(),thrust::not1(pred_A),688);
  checkVector(V9,CHECK133,__FILE__,__LINE__);
  thrust::replace_if(V4.begin()+1,V4.end()-1,thrust::not1(pred_C),-388);
  checkVector(V4,CHECK134,__FILE__,__LINE__);
  thrust::replace_if(V2.begin()+1,V2.end(),V7.begin(),pred_A,-28);
  checkVector(V2,CHECK135,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V6.begin(),V6.end()-1,thrust::not1(pred_D),-932);
  checkVector(V6,CHECK136,__FILE__,__LINE__);
  thrust::replace_if(V3.begin(),V3.end(),V9.begin(),pred_B,-608);
  checkVector(V3,CHECK137,__FILE__,__LINE__);
  thrust::replace_if(V2.begin(),V2.end()-1,V1.begin(),thrust::not1(pred_C),-453);
  checkVector(V2,CHECK138,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V6.begin(),V6.begin()+5,V1.begin(),pred_E,-20);
  checkVector(V1,CHECK139,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V3.begin(),V3.end()-1,V3.begin(),V1.begin(),thrust::not1(pred_A),-624);
  checkVector(V1,CHECK140,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V9.begin()+1,V9.end(),pred_B,691);
  checkVector(V9,CHECK141,__FILE__,__LINE__);
  thrust::replace_copy_if(V0.begin(),V0.end(),V7.begin(),pred_B,-210);
  checkVector(V7,CHECK142,__FILE__,__LINE__);
  thrust::replace_copy_if(V2.begin()+1,V2.end(),V7.begin(),pred_C,-487);
  checkVector(V7,CHECK143,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V6.begin(),V6.end(),V2.begin(),pred_A,818);
  checkVector(V6,CHECK144,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V6.begin(),V6.begin()+5,pred_D,-757);
  checkVector(V6,CHECK145,__FILE__,__LINE__);
  thrust::replace_copy_if(V9.begin()+1,V9.end(),V5.begin(),V8.begin(),pred_E,207);
  checkVector(V8,CHECK146,__FILE__,__LINE__);
  thrust::replace_copy_if(V1.begin()+1,V1.end()-1,V2.begin(),V9.begin(),pred_A,-251);
  checkVector(V9,CHECK147,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V5.begin(),V5.end(),V3.begin(),V8.begin(),pred_E,-767);
  checkVector(V8,CHECK148,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V9.begin()+1,V9.end()-1,V1.begin(),pred_C,-591);
  checkVector(V1,CHECK149,__FILE__,__LINE__);
  thrust::replace_copy_if(V6.begin(),V6.end()-1,V8.begin(),V7.begin(),thrust::not1(pred_E),762);
  checkVector(V7,CHECK150,__FILE__,__LINE__);
  thrust::replace_copy_if(V9.begin()+1,V9.end(),V6.begin(),V3.begin(),thrust::not1(pred_A),-213);
  checkVector(V3,CHECK151,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V0.begin()+1,V0.end(),pred_A,-857);
  checkVector(V0,CHECK152,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V9.begin()+1,V9.end(),V4.begin(),V3.begin(),thrust::not1(pred_E),-174);
  checkVector(V3,CHECK153,__FILE__,__LINE__);
  thrust::replace_if(V7.begin(),V7.end()-1,pred_A,70);
  checkVector(V7,CHECK154,__FILE__,__LINE__);
  thrust::replace_copy_if(V1.begin()+1,V1.end(),V8.begin(),pred_B,-842);
  checkVector(V8,CHECK155,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V1.begin(),V1.begin()+5,V4.begin(),V8.begin(),pred_B,37);
  checkVector(V8,CHECK156,__FILE__,__LINE__);
  thrust::replace_if(V9.begin(),V9.end(),pred_E,799);
  checkVector(V9,CHECK157,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V9.begin()+1,V9.end(),V5.begin(),V8.begin(),pred_C,-557);
  checkVector(V8,CHECK158,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V3.begin(),V3.begin()+5,V5.begin(),pred_C,32);
  checkVector(V3,CHECK159,__FILE__,__LINE__);
  thrust::replace_copy_if(V3.begin()+1,V3.end()-1,V2.begin(),V7.begin(),pred_D,438);
  checkVector(V7,CHECK160,__FILE__,__LINE__);
  thrust::replace_copy_if(V0.begin()+1,V0.end()-1,V3.begin(),V1.begin(),pred_E,342);
  checkVector(V1,CHECK161,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V0.begin(),V0.end()-1,V8.begin(),pred_E,254);
  checkVector(V0,CHECK162,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V8.begin(),V8.begin()+5,V0.begin(),V5.begin(),thrust::not1(pred_E),-361);
  checkVector(V5,CHECK163,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V4.begin()+1,V4.end(),V7.begin(),pred_B,-454);
  checkVector(V7,CHECK164,__FILE__,__LINE__);
  thrust::replace_if(V4.begin(),V4.begin()+5,pred_E,-158);
  checkVector(V4,CHECK165,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V9.begin(),V9.end(),V0.begin(),thrust::not1(pred_D),-732);
  checkVector(V9,CHECK166,__FILE__,__LINE__);
  thrust::replace_if(V9.begin()+1,V9.end()-1,pred_A,383);
  checkVector(V9,CHECK167,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V0.begin(),V0.begin()+5,V3.begin(),pred_D,703);
  checkVector(V3,CHECK168,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V3.begin()+1,V3.end()-1,V9.begin(),pred_C,18);
  checkVector(V3,CHECK169,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V0.begin()+1,V0.end(),V3.begin(),thrust::not1(pred_A),-394);
  checkVector(V0,CHECK170,__FILE__,__LINE__);
  thrust::replace_copy_if(V5.begin()+1,V5.end()-1,V4.begin(),V9.begin(),pred_C,-223);
  checkVector(V9,CHECK171,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V8.begin(),V8.begin()+5,V3.begin(),pred_E,-345);
  checkVector(V8,CHECK172,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V5.begin()+1,V5.end()-1,V2.begin(),pred_B,-180);
  checkVector(V5,CHECK173,__FILE__,__LINE__);
  thrust::replace_if(V8.begin(),V8.begin()+5,V0.begin(),pred_D,-226);
  checkVector(V8,CHECK174,__FILE__,__LINE__);
  thrust::replace_copy_if(V2.begin()+1,V2.end(),V7.begin(),V0.begin(),thrust::not1(pred_D),820);
  checkVector(V0,CHECK175,__FILE__,__LINE__);
  thrust::replace_copy_if(V8.begin(),V8.end()-1,V2.begin(),pred_D,-924);
  checkVector(V2,CHECK176,__FILE__,__LINE__);
  thrust::replace_copy_if(V6.begin()+1,V6.end()-1,V2.begin(),V5.begin(),pred_E,6);
  checkVector(V5,CHECK177,__FILE__,__LINE__);
  thrust::replace_copy_if(V0.begin()+1,V0.end(),V9.begin(),pred_B,916);
  checkVector(V9,CHECK178,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V3.begin(),V3.begin()+5,V1.begin(),V6.begin(),pred_C,-785);
  checkVector(V6,CHECK179,__FILE__,__LINE__);
  thrust::replace_if(V4.begin()+1,V4.end(),V5.begin(),thrust::not1(pred_C),-280);
  checkVector(V4,CHECK180,__FILE__,__LINE__);
  thrust::replace_copy_if(V2.begin(),V2.end()-1,V4.begin(),V3.begin(),thrust::not1(pred_C),847);
  checkVector(V3,CHECK181,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V5.begin(),V5.begin()+5,V3.begin(),pred_B,-85);
  checkVector(V3,CHECK182,__FILE__,__LINE__);
  thrust::replace_copy_if(V4.begin(),V4.begin()+5,V2.begin(),V1.begin(),thrust::not1(pred_D),570);
  checkVector(V1,CHECK183,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V0.begin(),V0.end()-1,V6.begin(),V2.begin(),pred_C,-309);
  checkVector(V2,CHECK184,__FILE__,__LINE__);
  thrust::replace_copy_if(V4.begin()+1,V4.end()-1,V6.begin(),pred_C,909);
  checkVector(V6,CHECK185,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V6.begin(),V6.begin()+5,V8.begin(),thrust::not1(pred_B),38);
  checkVector(V6,CHECK186,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V7.begin()+1,V7.end(),V2.begin(),V0.begin(),pred_E,821);
  checkVector(V0,CHECK187,__FILE__,__LINE__);
  thrust::replace_copy_if(V8.begin()+1,V8.end()-1,V4.begin(),V7.begin(),pred_D,108);
  checkVector(V7,CHECK188,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V0.begin(),V0.end(),V2.begin(),pred_A,-275);
  checkVector(V2,CHECK189,__FILE__,__LINE__);
  thrust::replace_if(V6.begin(),V6.begin()+5,pred_C,-609);
  checkVector(V6,CHECK190,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V1.begin(),V1.begin()+5,V0.begin(),thrust::not1(pred_E),538);
  checkVector(V1,CHECK191,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V0.begin(),V0.begin()+5,V9.begin(),V7.begin(),thrust::not1(pred_C),-856);
  checkVector(V7,CHECK192,__FILE__,__LINE__);
  thrust::replace_copy_if(V7.begin(),V7.begin()+5,V5.begin(),pred_E,410);
  checkVector(V5,CHECK193,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V3.begin(),V3.begin()+5,pred_A,751);
  checkVector(V3,CHECK194,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V6.begin(),V6.begin()+5,V0.begin(),pred_D,435);
  checkVector(V0,CHECK195,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V3.begin()+1,V3.end()-1,V8.begin(),V7.begin(),pred_D,828);
  checkVector(V7,CHECK196,__FILE__,__LINE__);
  thrust::replace_copy_if(thrust::host,V1.begin(),V1.end(),V0.begin(),pred_B,888);
  checkVector(V0,CHECK197,__FILE__,__LINE__);
  thrust::replace_if(V6.begin(),V6.begin()+5,V5.begin(),thrust::not1(pred_B),157);
  checkVector(V6,CHECK198,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V5.begin()+1,V5.end()-1,pred_A,486);
  checkVector(V5,CHECK199,__FILE__,__LINE__);
  thrust::replace_if(thrust::host,V1.begin()+1,V1.end()-1,V0.begin(),pred_A,-973);
  checkVector(V1,CHECK200,__FILE__,__LINE__);
  checkVector(V0,CHECK201,__FILE__,__LINE__);
  checkVector(V1,CHECK202,__FILE__,__LINE__);
  checkVector(V2,CHECK203,__FILE__,__LINE__);
  checkVector(V3,CHECK204,__FILE__,__LINE__);
  checkVector(V4,CHECK205,__FILE__,__LINE__);
  checkVector(V5,CHECK206,__FILE__,__LINE__);
  checkVector(V6,CHECK207,__FILE__,__LINE__);
  checkVector(V7,CHECK208,__FILE__,__LINE__);
  checkVector(V8,CHECK209,__FILE__,__LINE__);
  checkVector(V9,CHECK210,__FILE__,__LINE__);
#ifdef GENERATE_REFERENCE
  std::cout << "//SEED 1\n";
#else
  std::cout << "Passed seed=1\n";
#endif
  return 0;
}
