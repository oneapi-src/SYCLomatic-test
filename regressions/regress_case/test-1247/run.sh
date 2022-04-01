if [ -f compile_commands.json ]
then
rm compile_commands.json
fi
intercept-build /usr/bin/make
grep "\\\\" ./compile_commands.json 
if [ "x$?" != "x0" ]
then
exit 0
fi
exit -1
