#include <iostream>
#include <vector>

// give a vector A and a vector B, reorder A to match B

// exercise function for LU decomposition Swapping function
void reorderArrayToMatch(std::vector<int>& A, const std::vector<int>& B) {
    for (int i = 0; i < A.size(); ++i) {
        // when A[i] is not in the correct position
        if (A[i] != B[i]) {
            // find the index of the element that should be swapped with A[i]
            int swapIndex = -1;
            for (int j = 0; j < A.size(); ++j) {
                if (A[j] == B[i]) {
                    swapIndex = j;
                    break;
                }
            }

            // swap A[i] and A[swapIndex] to move A[i] to the correct position
            if (swapIndex != -1) {
                std::swap(A[i], A[swapIndex]);
            }
        }
    }
}

int main() {
    std::vector<int> A = {0, 1, 2};
    std::vector<int> B = {2,0,1};

    reorderArrayToMatch(A, B);

    // print the result
    for (int i = 0; i < A.size(); ++i) {
        std::cout << A[i] << " ";
    }

    return 0;
}
