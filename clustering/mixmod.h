#ifndef _MIXMOD_
#define _MIXMOD_

#include <mixmod/Kernel/IO/Data.h>
#include <mixmod/Kernel/IO/GaussianData.h>
#include <mixmod/Kernel/IO/DataDescription.h>
#include <mixmod/Clustering/ClusteringInput.h>
#include <mixmod/Clustering/ClusteringOutput.h>
#include <mixmod/Clustering/ClusteringModelOutput.h>
#include <mixmod/Clustering/ClusteringMain.h>

#include "clusters.h"
#include "../ematrix.h"


class MixMod {
  private:
    double ** data;
    int size;

    // The type of data used for this mixture model.
    XEM::DataType dataType;

    XEM::GaussianData * gdata;
    XEM::DataDescription * dataDescription;
    XEM::ClusteringInput * cInput;
    XEM::ClusteringOutput * cOutput;

    // nbCluster contains the numbers of clusters to be tested.
    vector<int64_t> nbCluster;
  public:
    MixMod(double *a, double *b, int n);
    ~MixMod();

    void run();
};

#endif
