int scale_main();
int sum_main();

int main() {
  if (scale_main()) return 1;
  if (sum_main())   return 1;

  return 0;
}
