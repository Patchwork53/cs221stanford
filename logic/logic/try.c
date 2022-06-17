#include<stdio.h>

using namespace std;

int sdbm( string str )
{
    int hash = 0;
    for( char ch: str )
    {
        hash = ch + (hash << 6) + (hash << 16) - hash;
        float x = 1.5-ABC;
    }
    return hash;
}

int main(){
    cout<<sdbm("foo")%7;
}