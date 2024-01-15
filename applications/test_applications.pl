use Cwd qw(cwd abs_path);
use File::Copy::Recursive qw(dircopy);
use File::Spec;
use File::Find;

use syclct qw(parse_option_set copy_open_source_dependent_files get_test_suite_name);

sub SetupTest {
  $canonical_test_suite_name = get_test_suite_name();
  if (!-d $canonical_test_suite_name) {
    copy_open_source_dependent_files();
  }
  return $PASS;
}
sub BuildTest {

  my @cmd = ("python3");
  push(@cmd, "run_test.py");
  push(@cmd, "-s " . $canonical_test_suite_name);
  push(@cmd, "-c " . $current_test);
  push(@cmd, "-o " . parse_option_set());
  execute(join(" ", @cmd));
  $execution_output .= $command_output;
  if ($command_status == 0) {
    return $PASS;
  }
  return $COMPFAIL;
}

sub RunTest {
  return $PASS
}

1;
