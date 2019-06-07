# Operating Systems

Operating systems sit between the hardware and user programs. There are 2 modes:

1. User Mode
2. Kernel Mode

The kernel mode allows for access to IO and performing some control ops. The user mode has a restricted instruction set available to it. Generally the user programs call into the kernel mode through a system call to perform some important operation. 

What an operating system actually does.

1. Provide application programmers a clean set of resources to develop on instead of messy hardware ones.
2. Manage the messy hardware resources.

Operating systems turn ugly hardware and its messy complicated usage patterns into beautiful abstractions that application programmers can use. Operating systems deal a lot with abstractions. 


# Processes

A process is informally a running program. 

HOw does the CPU run many processes at the same time, in order words how does the CPU give the illusion of several CPU's ?

THis illusion is called virtualising the CPU. IT does this by running a process, stopping it and running another at such a speed where it seems that the processes are running parallely. This technique is called **time sharing**. Although this will be at the *potential* cost of performance as each process willtake more time if the CPU must be shared.

To do virtualisation:
1. Low level machinery= **Mechanisms** eg: context switch
2. High Level intelligence = Policies Making intelligent decisions, eg: A scheduling policy for processes, given its historical usage, workload info, performance metrics. 


## The Abstraction

What des a process contain. TO define a process we need to understand it's machine state. It's machine state can be defined as 

1. What data is it currently storing ? (it's memory)
2. What instruction is it processing ?
3. 
TO  put it more succiciently the part of the memory that the process can address is called its **address space**

Also part of the machine sdtate are the registers, many instructions explicitly read and updater registers. There are some special registers that arte part of the machine state

1. Instruction/Program Pointer Register: whihc instruction of the program will execute next, 
2. Stack Pointer and frame poiner manage stack of function parameters, local vAriables and return addresses. 
3. Finally proceeses need to access IO devices, here I/O information might include a list of files the process has currently open.
4. 

## Process API

1. Create
2. Wait (OS waits for process to stop running)
3. Destroy
4. Pause/Resume
5. Status
6. 
To start the program to become a process, first the instruction set is loaded into memory from disk. The atck and heap are allocated. The stack with function parameters, return addresses and varibales declared on the stack. The heap is needed for dynamically allocated memory like datastructures. SOmetimes intructions/code is loaded lazily, through paging and swapping. 

Some IO file descriptors are opened, like i/o to write and read data from terminal. 

Then the main() function is called. 

