// Copyright (C) Rishabh Iyer, John T. Halloran, and Kai Wei
// Licensed under the Open Software License version 3.0
// See COPYING or http://opensource.org/licenses/OSL-3.0
#ifndef FIADMM_DENSE_FEATURE
#define FIADMM_DENSE_FEATURE

#include <vector>

namespace comlkit {

    struct DenseFeature { //Stores the feature vector for each item in the groundset
        long int index; // index of the item
        std::vector<double> featureVec; // score of the dense feature vector.
        int numFeatures;
    };
// This is a dense feature representation. Each row has numFeatures number of items (i.e the size of featureVec should be equal to numFeatures)

}

#endif
