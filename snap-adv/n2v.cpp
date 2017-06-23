#include "stdafx.h"
#include "n2v.h"

void node2vec(PWNet& InNet, double& ParamP, double& ParamQ, int& Dimensions,
 int& WalkLen, int& NumWalks, int& WinSize, int& Iter, bool& Verbose,
 TIntFltVH& EmbeddingsHV, double& NumRandomSample) {
  //Preprocess transition probabilities
  PreprocessTransitionProbs(InNet, ParamP, ParamQ, Verbose);
  TIntV NIdsV;
  for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
    NIdsV.Add(NI.GetId());
  }
  //printf("Finished preprocessing transition probabilities. Length of NIdsV is %d\n", NIdsV.Len());
  
  //Handle random sampling
  int len = 0;
  if (NumRandomSample > 0)
    len = NumRandomSample;
  else
    len = NIdsV.Len();

  //Generate random walks
  int64 AllWalks = (int64)NumWalks * len;
  TVVec<TInt, int64> WalksVV(AllWalks,WalkLen);
  TRnd Rnd(time(NULL));
  int64 WalksDone = 0;
  
  TIntV SelectedNodes(AllWalks);
  int t = 0;

  NIdsV.Shuffle(Rnd);
  for (int64 i = 0; i < len; i++) {
    SelectedNodes[t] = NIdsV[i];
    t += 1;
#pragma omp parallel for schedule(dynamic)
    for (int64 j = 0; j < NumWalks; j++) {
      TIntV WalkV;
      SimulateWalk(InNet, NIdsV[i], WalkLen, Rnd, WalkV);
      for (int64 k = 0; k < WalkV.Len(); k++) {
        WalksVV.PutXY(j * len + i, k, WalkV[k]);
      }
      WalksDone++;
    }
  }

  // TODO: Check if new logic replicates old logic
  
  /*
  for (int64 i = 0; i < NumWalks; i++) {
     NIdsV.Shuffle(Rnd);
#pragma omp parallel for schedule(dynamic)
    for (int64 j = 0; j < len; j++) {
      if ( Verbose && WalksDone%10000 == 0 ) {
        printf("\rWalking Progress: %.2lf%%",(double)WalksDone*100/(double)AllWalks);fflush(stdout);
      }
      TIntV WalkV;
      SimulateWalk(InNet, NIdsV[j], WalkLen, Rnd, WalkV);
      t += 1;
      printf("\t Added %d to SNodes\n", NIdsV[j]);
      SelectedNodes[t] = NIdsV[j];
      for (int64 k = 0; k < WalkV.Len(); k++) { 
        WalksVV.PutXY(i*len+j, k, WalkV[k]);
      }
      WalksDone++;
    }
  }
  */
  if (Verbose) {
    printf("\n");
    fflush(stdout);
  }
  //printf("WalksVV -> XDim :  %d\n", WalksVV.GetXDim());
  //printf("WalksVV -> YDim : %d\n", WalksVV.GetYDim());
  //printf("SNodes  -> Dim  : %d\n", SelectedNodes.Len());
  //Learning embeddings
  LearnEmbeddings(WalksVV, Dimensions, WinSize, Iter, Verbose, EmbeddingsHV, SelectedNodes);
  //printf("Embeddings -> (%d, %d)\n", EmbeddingsHV.GetXDim(), EmbeddingsHV.GetYDim());
}

void node2vec(PNGraph& InNet, double& ParamP, double& ParamQ, int& Dimensions,
 int& WalkLen, int& NumWalks, int& WinSize, int& Iter, bool& Verbose,
 TIntFltVH& EmbeddingsHV, double& NumRandomSample) {
  PWNet NewNet = PWNet::New();
  for (TNGraph::TEdgeI EI = InNet->BegEI(); EI < InNet->EndEI(); EI++) {
    if (!NewNet->IsNode(EI.GetSrcNId())) { NewNet->AddNode(EI.GetSrcNId()); }
    if (!NewNet->IsNode(EI.GetDstNId())) { NewNet->AddNode(EI.GetDstNId()); }
    NewNet->AddEdge(EI.GetSrcNId(), EI.GetDstNId(), 1.0);
  }
  node2vec(NewNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize, Iter, 
   Verbose, EmbeddingsHV, NumRandomSample);
}

void node2vec(PNEANet& InNet, double& ParamP, double& ParamQ,
 int& Dimensions, int& WalkLen, int& NumWalks, int& WinSize, int& Iter, bool& Verbose,
 TIntFltVH& EmbeddingsHV, double& NumRandomSample) {
  PWNet NewNet = PWNet::New();
  for (TNEANet::TEdgeI EI = InNet->BegEI(); EI < InNet->EndEI(); EI++) {
    if (!NewNet->IsNode(EI.GetSrcNId())) { NewNet->AddNode(EI.GetSrcNId()); }
    if (!NewNet->IsNode(EI.GetDstNId())) { NewNet->AddNode(EI.GetDstNId()); }
    NewNet->AddEdge(EI.GetSrcNId(), EI.GetDstNId(), InNet->GetFltAttrDatE(EI,"weight"));
  }
  node2vec(NewNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize, Iter, 
   Verbose, EmbeddingsHV, NumRandomSample);
}
