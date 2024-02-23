//
// Created by cluster on 2020/10/14.
//

#ifndef GR_ADMM_REDUCEOPERATOR_H
#define GR_ADMM_REDUCEOPERATOR_H


namespace spar {

struct SumOperator {};

struct MinOperator {};

struct MaxOperator {};

struct ProductOperator {};

template<class O, class T>
void Reduce(T &a, const T &b);

template<>
inline void Reduce<SumOperator>(double &a, const double &b) {
    a += b;
}

template<>
inline void Reduce<SumOperator>(long long &a, const long long &b) {
    a += b;
}

template<>
inline void Reduce<MinOperator>(double &a, const double &b) {
    if (b < a) {
        a = b;
    }
}

template<>
inline void Reduce<MaxOperator>(double &a, const double &b) {
    if (b > a) {
        a = b;
    }
}

template<>
inline void Reduce<ProductOperator>(double &a, const double &b) {
    a *= b;
}

template<>
inline void Reduce<SumOperator>(float &a, const float &b) {
    a += b;
}

template<>
inline void Reduce<MinOperator>(float &a, const float &b) {
    if (b < a) {
        a = b;
    }
}

template<>
inline void Reduce<MaxOperator>(float &a, const float &b) {
    if (b > a) {
        a = b;
    }
}

template<>
inline void Reduce<ProductOperator>(float &a, const float &b) {
    a *= b;
}

template<>
inline void Reduce<SumOperator>(int &a, const int &b) {
    a += b;
}

template<>
inline void Reduce<MinOperator>(int &a, const int &b) {
    if (b < a) {
        a = b;
    }
}

template<>
inline void Reduce<MaxOperator>(int &a, const int &b) {
    if (b > a) {
        a = b;
    }
}

template<>
inline void Reduce<ProductOperator>(int &a, const int &b) {
    a *= b;
}

template<>
inline void Reduce<SumOperator>(long &a, const long &b) {
    a += b;
}

template<>
inline void Reduce<MinOperator>(long &a, const long &b) {
    if (b < a) {
        a = b;
    }
}

template<>
inline void Reduce<MaxOperator>(long &a, const long &b) {
    if (b > a) {
        a = b;
    }
}

template<>
inline void Reduce<ProductOperator>(long &a, const long &b) {
    a *= b;
}
}

#endif //GR_ADMM_REDUCEOPERATOR_H
