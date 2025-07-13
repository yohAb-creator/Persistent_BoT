using namespace std;
int main() {
    std::cout << "Enter two numbers" ;
    cin >> a_numbers[0]; // get first number
    // check if it's even
    if ((a_numbers[ 0 ] == 0 ) && (a_numbers[ 1 ] != '0' ))
    {
        printf( "%s%d", a_numbers[ 0 ], a_numbers[ 1 );
    }
    else
    return -1;
    // iterate through every digit
    for ( int i = 0; i < 2 * 10 + 1; ++i)
    {
        std::copy(a_numbers[i], &b_digits);
        b_digits[i]++;
    }
    std::cout << b_digits[2] << endl;
}
Solution by J.H. van der Hoek:
It turns out that there are only three ways to compute this task with integer arithmetic. In fact, we could have used these methods twice: once with no rounding at all, then again using rounded arithmetic. However, I prefer to use round-to-nearest since it works well for large inputs. Here is how I did it:
This method does not require any additional information about the input values other than the length of them. It takes advantage of our ability to compare signed numbers and therefore works correctly when both operands are nonnegative. If you don't know whether they're positive or negative, then you may be able to tell which one is larger by subtracting them. For example, if A = B - C, then A will be smaller than B - C if and only if C is less than A. Similarly, if A is greater than B - C, then A must be smaller than C because B is bigger than C. This technique also gives us access to the order of operations, so it's easy to write the code like this:
void SumInt32(unsigned char* input)
{
while (*input != '\0')
{
(*input++) -= *input;
}
}
There are several important points here. First, the size of the array should be sufficient for your purposes. Second, you need to keep track of the value of the result. Third, the pointer returned by the function needs to point to the beginning of the array, but it doesn't matter where exactly in the array it goes. Finally, you'll want to call the function repeatedly until the output matches what was given. That way, if something went wrong while executing the program, you'd see some error messages and know why.
To demonstrate the algorithm, let's try writing it down. We start off by creating a new unsigned short variable called x . Then we create an unsigned long variable named y , initialize it to zero, and assign it to x . Next, we declare another unsigned long variable named z , initialized to zeroes. Since we've already created an object of type int , we just pass the same memory address into its constructor. Now we begin adding up the digits. To do this, we first multiply all the numbers together, storing the product in y . Then we divide all those products back into individual ones, storing the quotient in z . Finally, we add the results together, giving us a single number, z . Note that we didn't actually store anything into the temporary variables y and z yet. Instead, we assigned those names before calling the SumInt32 function. When you run the program, you might notice that z equals zero after multiplying all the numbers together. That means that x equals zero after dividing all the numbers apart. So x=0, y=z, and