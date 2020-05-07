# vs-kangaroo-hybrid
Forked from https://github.com/Telariust/vs-kangaroo from https://github.com/JeanLucPons/VanitySearch from https://github.com/JeanLucPons/Kangaroo 

This experimental project.

# Discussion

https://bitcointalk.org/index.php?topic=5218972.msg54083292#msg54083292

# Usage

./vs-kangaroo-hybrid -v 1 -gpu -d 14 -bits 65 0230210c23b1a047bc9bdbb13448e67deddc108946de6de639bcc75d47c0216b1b 

./vs-kangaroo-hybrid -v 1 -gpu -d 14 -bits 75 03726b574f193e374686d8e12bc6e4142adeb06770e0a2856f5e4ad89f66044755 

----------------------------------------------------------------------------------------------------------------

Check Bits: 110
Compressed Address: 1FZgkq1dqCfgxaCFVkoP9sZuFFja1FTwMP
Address hash160: 9fc042129e3c1c5649beee5f575babfc85eba9db
Secret wif: KwDiBf89QgGbjEhKnhXJuH7LrctqV9Hyh5UagDz41NsKpRbQHnAx
Secret hex: 0x3878efc360a7e843c1517c415f07L
pk: 0352c323c1fe80131546a81141657cbd7f84a00a869cdb31c3b09310048bdc2077

./vs-kangaroo-hybrid -v 1 -d 14 -gpu -bits 3878efc360a6e843c1517c415f07:3878efc360a8e843c1517c415f07 0352c323c1fe80131546a81141657cbd7f84a00a869cdb31c3b09310048bdc2077 


Check Bits: 256
Compressed Address: 1LrqA7saB8tP7NN42yDui4Y2VxQkQKGnTL
Address hash160: d9d6f8bd48f1056c0413d21979774348573f8981
Secret wif: L2iAxC8cAhWHk435CpmQ6CAxLWYXpryCXctYVXEJ5NikpfaCucCf
Secret hex: 0xa3d4235179bd55c0ba6595fbef8cd2a45efa5fd31ce25150eb9b72b8ff810cb3L
pk: 02be96c2c11b385b20e5783ded59293a6e7e714d99e259d99cd1e6071332834324

./vs-kangaroo-hybrid -v 1 -d 14 -gpu -bits a3d4235179bd55c0ba6595fbef8cd2a45efa5fd31ce25150eb9a72b8ff810cb3:a3d4235179bd55c0ba6595fbef8cd2a45efa5fd31ce25150eb9c72b8ff810cb3 02be96c2c11b385b20e5783ded59293a6e7e714d99e259d99cd1e6071332834324 


----------------------------------------------------------------------------------------------------------------

# Compilation

## Windows

Intall CUDA SDK and open vs-kangaroo.sln in Visual C++ 2017.\
You may need to reset your *Windows SDK version* in project properties.\
In Build->Configuration Manager, select the *Release* configuration.\
Build and enjoy.\
\
Note: The current relase has been compiled with CUDA SDK 10.2, if you have a different release of the CUDA SDK, you may need to update CUDA SDK paths in vs-kangaroo.vcxproj using a text editor. The current nvcc option are set up to architecture starting at 3.0 capability, for older hardware, add the desired compute capabilities to the list in GPUEngine.cu properties, CUDA C/C++, Device, Code Generation.

## Linux

Intall CUDA SDK.\
Depenging on the CUDA SDK version and on your Linux distribution you may need to install an older g++ (just for the CUDA SDK).\
Edit the makefile and set up the good CUDA SDK path and appropriate compiler for nvcc. 

```
CUDA       = /usr/local/cuda-8.0
CXXCUDA    = /usr/bin/g++-4.8
```

You can enter a list of architectrure (refer to nvcc documentation) if you have several GPU with different architecture. Compute capability 2.0 (Fermi) is deprecated for recent CUDA SDK.
vs-kangaroo-hybrid need to be compiled and linked with a recent gcc (>=7). The current release has been compiled with gcc 7.3.0.\
Go to the vs-kangaroo-hybrid directory. 
ccap is the desired compute capability https://ru.wikipedia.org/wiki/CUDA

```
$ g++ -v
gcc version 7.3.0 (Ubuntu 7.3.0-27ubuntu1~18.04)
$ make all (for build without CUDA support)
or
$ make gpu=1 ccap=20 all
```

# License

vs-kangaroo-hybrid is licensed under GPLv3.
