# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name: str):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


class Solution:
    def quickSort(self, nums: List[int], l:int, r:int):
        if l<r:
            mid = self.partition(nums, l, r)
            self.quickSort(nums, l, mid-1)
            self.quickSort(nums, mid+1, r)

    def partition(self, nums:List[int], l:int, r:int) ->int:
        m = nums[l]
        i = l
        j = l+1
        while j <= r:
            if nums[j] < m:
                i += 1
                self.swap(nums,i,j)
            j += 1
        self.swap(nums,l,i)
        return l

    def swap(self,nums,i,j):
        temp = nums[i]
        nums[i] = nums[j]
        nums[j] = temp


    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) <2:
            return nums
        self.quickSort(nums,0,len(nums)-1)
        return nums

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


