#include <vector>
#include <cstdio>
#include <iostream>

using namespace std;

typedef struct {
	float value;
	unsigned count;
	unsigned average;
} thresholdValue_t;

int main(int argc, char const *argv[])
{
    char const* const fileName = argv[1]; /* should check that argc > 1 */
    FILE* file = fopen(fileName, "r"); /* should check the result */
    char line[256];


    if (NULL == database)
    {
         perror("opening database");
         return (-1);
    }
    unsigned num = 0;
    float gamma = 0;

    vector<thresholdValue_t> threshold;
    while (EOF != fscanf(database, "%u,%f,%f,%u\n", num))
    {
         printf("> %s\n", buffer);
    }

    fclose(database);
    return (0);
}