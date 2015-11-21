<?php
echo getcwd() . "\n";
chdir('/etc');
echo getcwd() . "\n";

$output = shell_exec('env');
echo "<pre>$output</pre>";

var_dump($_ENV)
?>
