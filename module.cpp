#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>

// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) {
    return tensor[x * (sizeX*sizeY*sizeZ) + y * (sizeY*sizeZ) + z * (sizeZ) + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
    tensor[x * (sizeX*sizeY*sizeZ) + y * (sizeY*sizeZ) + z * (sizeZ) + b] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int qi = 0; qi < N; qi++) {
                for (int ki = 0; ki < N; ki++) {
                    float val = 0;
                    for (int k = 0; k < d; k++) {
                        float q_val = fourDimRead(Q, b, h, qi, k, H, N, d);
                        float k_val = fourDimRead(K, b, h, ki, k, H, N, d);
                        val += q_val * k_val;
                    }
                    twoDimWrite(QK_t, qi, ki, N, val);
                }
            }

            for (int i = 0; i < N; i++) {
                float expsum = 0;
                for (int j = 0; j < N; j++) {
                    float val = twoDimRead(QK_t, i, j, N);
                    val = exp(val);
                    expsum += val;
                    twoDimWrite(QK_t, i, j, N, val);
                }
                for (int j = 0; j < N; j++) {
                    float val = twoDimRead(QK_t, i, j, N);
                    val /= expsum;
                    twoDimWrite(QK_t, i, j, N, val);
                }
            }

            for (int qki = 0; qki < N; qki++) {
                for (int vj = 0; vj < d; vj++) {
                    float val = 0;
                    for (int k = 0; k < N; k++) {
                        float qk_val = twoDimRead(QK_t, qki, k, N);
                        float v_val = fourDimRead(V, b, h, k, vj, H, N, d);
                        val += qk_val * v_val;
                    }
                    fourDimWrite(O, b, h, qki, vj, H, N, d, val);
                }
            }
        }
    }

    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //
    const int TILE_SIZE = 16;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int qi = 0; qi < N; qi += TILE_SIZE) {
                for (int ki = 0; ki < N; ki += TILE_SIZE) {

                    // init to zero before dot product
                    for (int ti = qi; ti < std::min(qi + TILE_SIZE, N); ti++) {
                        for (int tj = ki; tj < std::min(ki + TILE_SIZE, N); tj++) {
                            float zero = 0;
                            twoDimWrite(QK_t, ti, tj, N, zero);
                        }
                    }

                    for (int k = 0; k < d; k += TILE_SIZE) {
                        for (int ti = qi; ti < std::min(qi + TILE_SIZE, N); ti++) {
                            for (int tj = ki; tj < std::min(ki + TILE_SIZE, N); tj++) {

                                float val = twoDimRead(QK_t, ti, tj, N);
                                for (int tk = k; tk < std::min(k + TILE_SIZE, d); tk++) {
                                    float q_val = fourDimRead(Q, b, h, ti, tk, H, N, d);
                                    float k_val = fourDimRead(K, b, h, tj, tk, H, N, d);
                                    val += q_val * k_val;
                                }
                                twoDimWrite(QK_t, ti, tj, N, val);

                            }
                        }
                    }

                }
            }

            for (int i = 0; i < N; i++) {
                float sum = 0;
                for (int j = 0; j < N; j++) {
                    float val = twoDimRead(QK_t, i, j, N);
                    val = exp(val);
                    sum += val;
                    twoDimWrite(QK_t, i, j, N, val);
                }
                for (int j = 0; j < N; j++) {
                    float val = twoDimRead(QK_t, i, j, N);
                    val /= sum;
                    twoDimWrite(QK_t, i, j, N, val);
                }
            }

            for (int qki = 0; qki < N; qki += TILE_SIZE) {
                for (int vj = 0; vj < d; vj += TILE_SIZE) {
                    for (int k = 0; k < N; k += TILE_SIZE) {
                        for (int ti = qki; ti < std::min(qki + TILE_SIZE, N); ti++) {
                            for (int tj = vj; tj < std::min(vj + TILE_SIZE, d); tj++) {

                                float val = fourDimRead(O, b, h, ti, tj, H, N, d);
                                for (int tk = k; tk < std::min(k + TILE_SIZE, N); tk++) {
                                    float qk_val = twoDimRead(QK_t, ti, tk, N);
                                    float v_val = fourDimRead(V, b, h, tk, tj, H, N, d);
                                    val += qk_val * v_val;
                                }
                                fourDimWrite(O, b, h, ti, tj, H, N, d, val);

                            }
                        }
                    }
                }
            }
        }
    }

    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int qi = 0; qi < N ; qi++) {
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
                std::vector<float> ORow = formatTensor(ORowTensor);
                
                float expsum = 0;
                for (int ki = 0; ki < N; ki++) {
                    float val = 0;
                    for (int k = 0; k < d; k++) {
                        float q_val = fourDimRead(Q, b, h, qi, k, H, N, d);
                        float k_val = fourDimRead(K, b, h, ki, k, H, N, d);
                        val += q_val * k_val;
                    }
                    val = exp(val);
                    expsum += val;
                    ORow[ki] = val;
                }

                for (int vj = 0; vj < d; vj++) {
                    float val = 0;
                    for (int k = 0; k < N; k++) {
                        float o_val = ORow[k] / expsum;
                        float v_val = fourDimRead(V, b, h, k, vj, H, N, d);
                        val += o_val * v_val;
                    }
                    fourDimWrite(O, b, h, qi, vj, H, N, d, val);
                }
            }
        }
    }
	
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // init l to zero
            for (int i = 0; i < N; i++) {
                l[i] = 0;
            }
            for (int ki = 0; ki < N; ki += Bc) {
                const int K_TILE_SIZE = std::min(Bc, N - ki);
                
                // load K, V
                for (int i = 0; i < K_TILE_SIZE; i++) {
                    int ki_tile = ki + i;
                    for (int j = 0; j < d; j++) {
                        float k_val = fourDimRead(K, b, h, ki_tile, j, H, N, d);
                        twoDimWrite(Kj, i, j, d, k_val);
                        float v_val = fourDimRead(V, b, h, ki_tile, j, H, N, d);
                        twoDimWrite(Vj, i, j, d, v_val);
                    }
                }

                for (int qi = 0; qi < N; qi += Br) {
                    const int Q_TILE_SIZE = std::min(Br, N - qi);

                    // load Q, O, l
                    for (int i = 0; i < Q_TILE_SIZE; i++) {
                        int qi_tile = qi + i;
                        for (int j = 0; j < d; j++) {
                            float q_val = fourDimRead(Q, b, h, qi_tile, j, H, N, d);
                            twoDimWrite(Qi, i, j, d, q_val);
                            float o_val = fourDimRead(O, b, h, qi_tile, j, H, N, d);
                            twoDimWrite(Oi, i, j, d, o_val);
                        }
                        float l_val = l[qi_tile];
                        li[i] = l_val;
                    }

                    // S = QK^T
                    for (int si = 0; si < Q_TILE_SIZE; si++) {
                        for (int sj = 0; sj < K_TILE_SIZE; sj++) {

                            float val = 0;
                            for (int k = 0; k < d; k++) {
                                float q_val = twoDimRead(Qi, si, k, d);
                                float k_val = twoDimRead(Kj, sj, k, d);
                                val += q_val * k_val;
                            }
                            twoDimWrite(Sij, si, sj, Bc, val);

                        }
                    }

                    // P = exp(S)
                    for (int si = 0; si < Q_TILE_SIZE; si++) {
                        for (int sj = 0; sj < K_TILE_SIZE; sj++) {
                            float val = twoDimRead(Sij, si, sj, Bc);
                            val = exp(val);
                            twoDimWrite(Pij, si, sj, Bc, val);
                        }
                    }
                            
                    // l = rowsum(P)
                    for (int si = 0; si < Q_TILE_SIZE; si++) {
                        float rowsum = 0;
                        for (int sj = 0; sj < K_TILE_SIZE; sj++) {
                            float val = twoDimRead(Pij, si, sj, Bc);
                            rowsum += val;
                        }
                        lij[si] = rowsum;
                    }

                    // update l
                    for (int si = 0; si < Q_TILE_SIZE; si++) {
                        lnew[si] = li[si] + lij[si];
                    }

                    // PV
                    for (int pi = 0; pi < Q_TILE_SIZE; pi++) {
                        for (int vj = 0; vj < d; vj++) {
                            float val = 0;
                            for (int k = 0; k < K_TILE_SIZE; k++) {
                                float p_val = twoDimRead(Pij, pi, k, Bc);
                                float v_val = twoDimRead(Vj, k, vj, d);
                                val += p_val * v_val;
                            }
                            twoDimWrite(PV, pi, vj, d, val);
                        }
                    }

                    // update O
                    for (int pi = 0; pi < Q_TILE_SIZE; pi++) {
                        for (int vj = 0; vj < d; vj++) {
                            float pv_val = twoDimRead(PV, pi, vj, d);
                            float o_val = twoDimRead(Oi, pi, vj, d);
                            o_val = (li[pi] * o_val + pv_val) / lnew[pi];
                            twoDimWrite(Oi, pi, vj, d, o_val);
                        }
                    }
                    
                    // update memory
                    for (int i = 0; i < Q_TILE_SIZE; i++) {
                        int qi_tile = qi + i;
                        for (int j = 0; j < d; j++) {
                            float o_val = twoDimRead(Oi, i, j, d);
                            fourDimWrite(O, b, h, qi_tile, j, H, N, d, o_val);
                        }
                        float l_val = lnew[i];
                        l[qi_tile] = l_val;
                    }
                }   
            }
        }
    }


    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
