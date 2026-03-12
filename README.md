# Labs 2026

<details>
<summary>Lab 1</summary>

## Lab 1

Soon
</details>

<details>
<summary>Lab 2</summary>

## Lab 2

Soon
</details>

<details>
<summary>Lab 3</summary>

## Lab 3

<details>
<summary>C++ quick reminder on pointers / arrays</summary>

### C++ quick reminder on pointers / arrays

<details>
<summary>I. Old way with pointers</summary>

#### I. The *OLD* way of returning arrays with pointers

```cpp
#include <iostream>

using namespace std;

int* getNumbers() {
    int* p = new int[3]{10, 20, 30};
    return p;
}

int main() {
  int* p = getNumbers();
  
  // use p
  for (int i=0; i<3; i++) {
      cout << p[i] << endl;
  }
  
  delete[] p;
}
```

Result:
```
10
20
30
```

The issue with this is that:
- easy to forget delete[]
-	memory leaks
- unclear ownership
- array size is not included

This is usually recommended in special cases where you need explicit control of memory, such as high-reliability systems

</details>

<details>
<summary>II. DYNAMIC size arrays</summary>

#### II. Using `std::vector<type>` for *DYNAMIC* size arrays

In case you would like to use arrays for whatever task, the simplest way is with `std::vector<>`

```cpp
#include <iostream>
#include <vector> // DON'T FORGET TO INCLUDE THIS!

using namespace std;

std::vector<int> getNumbers() {
    return {10, 20, 30};    
}

int main() {
  std::vector<int> p = getNumbers();
  
  // use p WITH range based loops
  for (int element: p) {
    cout << element << endl;
  }
  
  cout << "---" << endl;
  
  // using with regular loops
  for (int i=0; i<p.size(); i++) {
    cout << p[i] << endl;
  }
  
  cout << "---" << endl;
  
  // Specifying size of the array, but all values are 0
  // I would need to specify the value of each element separatly to change it
  // another_vector[0] = 10;
  
  std::vector<int> another_vector(3);
    
  for (int element: another_vector) {
    cout << element << endl;
  }
}
```

Result:
```
10
20
30
---
10
20
30
---
0
0
0
```

</details>

<details>
<summary>III. STATIC size arrays</summary>

#### III. Using `std:array<type, size>` for *STATIC* size arrays

In case I know the size of the array at compile time, then it's better to use `std::array<type, size>`

```cpp
#include <iostream>
#include <array> // DON'T FORGET TO INCLUDE THIS!

using namespace std;

std::array<int, 3> getNumbers() {
    return {10, 20, 30};    
}

int main() {
  std::array<int, 3> p = getNumbers();
  
  // use p WITH range based loops
  for (int element: p) {
    cout << element << endl;
  }
  
  cout << "---" << endl;
  
  // using with regular loops
  for (int i=0; i<p.size(); i++) {
    cout << p[i] << endl;
  }  
}
```

</details>

</details>

</details>
