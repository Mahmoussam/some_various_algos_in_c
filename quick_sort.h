/*
    by Unknown Nine Tailed Fox solder
    Quicksort Lomuto scheme.
    steps:
    partition(arr[], lo, hi) 
        pivot = arr[hi]
        i = lo-1     // place for swapping
        for j := lo to hi – 1 do
            if arr[j] <= pivot then
                i = i + 1 
                swap arr[i] with arr[j]
                print whole array
        swap arr[i+1] with arr[hi]
        print whole array
        partition(arr[],lo,i)
        partition(arr[],i+2,hi)
*/

#include <stdio.h>
#include <stdlib.h>

void quick_sort(int *array, size_t size);
void partition(int *array,int lo,int hi,size_t size);
void print_array(int *array,size_t size);
void swap(int *p1,int *p2);
void swap(int *p1,int *p2){
    int tmp=*p1;
    *p1=*p2;
    *p2=tmp;
}
void quick_sort(int *array,size_t size){
    partition(array,0,size-1,size);
}

void partition(int *array,int lo,int hi,size_t size){
    if(lo>=hi)return;
    int pivot=array[hi];
    int i=lo-1;
    for(int j=lo;j<hi;j++){
        if(array[j]<=pivot){
            i++;
            swap(&array[i],&array[j]);
            print_array(array,size);
        }
    }
    
    swap(&array[i+1],&array[hi]);
    print_array(array,size);
    partition(array,lo,i,size);
    partition(array,i+2,hi,size);
}
void print_array(int *array,size_t size){
    for(int i=0;i<size;i++){
        printf("%d ",array[i]);
    }
    printf("\n");
}

