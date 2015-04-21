#include <stdio.h>
#include <stdlib.h>

int result(int input[], double weights[]) {

   double in_sum = 0;
   int i;
   
   for (i = 0; i <= 2; i++) {
      in_sum = in_sum + input[i] * weights[i];
   }
   if (in_sum >= 0) {
      return 1;
   }
   return 0;
   
}

int main(void) {
   
   double alpha = 1;
   double weights[] = {-0.5, -0.5, -0.5};
   int epochs = 0;
   int CORRECT = 0;
   double delta;

   int input[4][3] = {{-1,1,1},{-1,1,0},{-1,0,1},{-1,0,0}};
   int target_output[] = {1, 0, 0, 0};

   int example;
   int i;
   
   while (epochs < 10) {
      
      /* Training while loop */
      example = 0;
      while (example < 4) {
         delta = target_output[example] - result(input[example], weights);
         for (i = 0; i <= 2; i++) {
            weights[i] = weights[i] + alpha * delta * input[example][i];
         }
         example++;
        
      }

      /* Testing while loop */
      example = 0;
      while (example < 4) {
         if (result(input[example], weights) == target_output[example]) { CORRECT++; }
         
         example++;
      }
      
      printf("%d\n",CORRECT);
      
      if (CORRECT == 4) {
         printf("Done!\n");
         break;
      }
      else {
         CORRECT = 0;
      }
      epochs++;
   }
   return EXIT_SUCCESS;

}
