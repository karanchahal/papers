# _Go_ : A Tour of Concurrency

 

Before diving into Go Concurrency. Let's talk about the Go language for a bit.

 

Why would you want to use Go ?

Mainly because it's got first class concurrency support. Go is a compiled language and rivals C/C++ in speed. It has a lot of modern features and has a very easy learning curve.

 

Softwares using Go in their stack.

 

1. Docker

2. Kubernetes

3. Ethereum

4. InfluxDB

 

``` Go is the server language of the future ```

 

## _Go_ Features

 

1. Concurrency and syncronisation support in the form of go routines

2. Go routines very cheap, like a lightweight thread. Only 4 kb compared to 1 MB taken up by the thread on the heap.

3. Can fire 1000s of go routiunes in a thread.

4. A good language to use the capabilities of modern multicore processors.

5. Compiled language , quite fast

6. Handles memory, no need for malloc() or free(). The Go garbage collector handles this.

 

# _Go_ Need to Knows

 

These are a few things that one needs to know about Go. This could be a plus or minus for you depending on what kind of programmer you are.

 

1. No inheritance

2. No classes

3. Only structs

4. No constructers.

5. No generics

6. No exceptions

 

The thought process behind removing these is to reduce langauge blot. Also the writers profess that you can write really clean maintainable code without these abstractions.

 

 

# _Go_ Concurrency

 

Now, let's get to the part you all have been waiting for. How do you make programs concurrent?

 

First what is concurrency.

In modern conmputers, one process doesn't work all the time. The CPU's process scheduler gives a little bit of time to each process in a circular fashion until all the processes are completed. Concurrent programs make use of this architecture by dividing a single program execution into various such processes. Hence, more of the program runs in less amount of time.

 

When two processes run in parallel on different cores, we achieve parallelism. Running various processes in a single CPU is called concurrency.

 

``` Concurrency is not parallelism ```

 

 

 

Let's get into it. This article will be divided into 2 parts.

1. Go Concurrency tools

2. Go Concurrency Patterns

 

## _Go_ Concurrency Tools

 

 

Go supports two primitives for Concurrency.

 

### Go Routines

 

You can run any function in its own go routine using the keyword go in front of your function call.

 

``` go testFunc() ```

 

### Go Channels

 

We need a way to communicate with other go routines. To do this, we use go channels.

 

Channels are a way to send/recieve information. Yoiu can make a channel and everyone subscribing to that channel recives information via that channel.

 

```

c := make(chan int)

```

 

This makes a channel named c which passes along int values.

 

```

c <- 5

x <- c // x in a int variable

```

 

The ```<-``` or ```->``` decided which direction to send some data through the channel.

 

Example Program using go routines and channels.

 

```

func boring(msg string, c chan string) {

    for i := 0; ; i++ {

        c <- fmt.Sprintf("%s %d", msg, i) // Expression to be sent can be any suitable value.

        time.Sleep(time.Duration(rand.Intn(1e3)) * time.Millisecond)

    }

}

 

func main() {

    c := make(chan string)

    go boring("boring!", c)

    for i := 0; i < 5; i++ {

        fmt.Printf("You say: %q\n", <-c) // Receive expression is just a value.

    }

    fmt.Println("You're boring; I'm leaving.")

}

 

```

 

Note that one has to transfer the channel in the argument of the function to be able to communicate.

 

 

## _Go_ Patterns

 

Channels are first class values , just like Ints and Strings.

 

### Generator Pattern

 

This function returns a channel , while running the respective function

 

```

 

c := boring("boring!") // Function returning a channel.

    for i := 0; i < 5; i++ {

        fmt.Printf("You say: %q\n", <-c)

    }

    fmt.Println("You're boring; I'm leaving.")

 

 

func boring(msg string) <-chan string { // Returns receive-only channel of strings.

    c := make(chan string)

    go func() { // We launch the goroutine from inside the function.

        for i := 0; ; i++ {

            c <- fmt.Sprintf("%s %d", msg, i)

            time.Sleep(time.Duration(rand.Intn(1e3)) * time.Millisecond)

        }

    }()

    return c // Return the channel to the caller.

}

 

```

 

We can use the generator pattern to communicate with different instances of a service.

 

```

func main() {

    joe := boring("Joe")

    ann := boring("Ann")

    for i := 0; i < 5; i++ {

        fmt.Println(<-joe)

        fmt.Println(<-ann)

    }

    fmt.Println("You're both boring; I'm leaving.")

}

 

```

 

### Multiplexing

 

In the previous program we make Joe and Ann count in lockstep. They are not independent of each other.

 

We can have something called a _fan in_ function to let whosoever is ready to talk.

 

```

func fanIn(input1, input2 <-chan string) <-chan string {

    c := make(chan string)

    go func() { for { c <- <-input1 } }()

    go func() { for { c <- <-input2 } }()

    return c

}

 

func main() {

    c := fanIn(boring("Joe"), boring("Ann"))

    for i := 0; i < 10; i++ {

        fmt.Println(<-c)

    }

    fmt.Println("You're both boring; I'm leaving.")

}

 

```

 

Through this method we can get various outputs from multiple go routines without blocking anything.

 

 

### Restoring Sequencing

 

????

 

### Select Statement

 

The select statement is a switch case, but every case is a communication via a channel.

 

- All channels are evaluated

- Selection blocks everything else until one communicatoin can proceed.

- If multiple channels can proceed, we pick anyone pseudo randomly

- A default clause if present , executed immediately if no channel is ready.

 

 

```

 

select {

    case v1 := <-c1:

        fmt.Printf("received %v from c1\n", v1)

    case v2 := <-c2:

        fmt.Printf("received %v from c2\n", v1)

    case c3 <- 23:

        fmt.Printf("sent %v to c3\n", 23)

    default:

        fmt.Printf("no one was ready to communicate\n")

    }

 

```

 

 

### Rewite Fan In

 

Now using the patterns that we have right now, we can rewrite our original Fan In.

 

```

func fanIn(input1, input2 <-chan string) <-chan string {

    c := make(chan string)

    go func() {

        for {

            select {

            case s := <-input1:  c <- s

            case s := <-input2:  c <- s

            }

        }

    }()

    return c

}

 

```

This is more efficient as we just have one function instead of two.

 

### Incorporating Time Out From Fan In

 

We can include a time out, if a channel  has no resposne for a while , we can do something to time out. Here go's time.After

 

```

func main() {

    c := boring("Joe")

    for {

        select {

        case s := <-c:

            fmt.Println(s)

        case <-time.After(1 * time.Second):

            fmt.Println("You're too slow.")

            return

        }

    }

}

 

```

 

We can create timeout for the entire conversatoin using a timeAfter outside the for loop.  Hence, we can define how lon we want the total conversation to carry on for.

 

## Telling the Channel to Quit

 

We can turn this around and tell the channel that we are tired of listening to him.

 

```

 

quit := make(chan bool)

    c := boring("Joe", quit)

    for i := rand.Intn(10); i >= 0; i-- { fmt.Println(<-c) }

    quit <- true

 

```

```

// in channel function now

            select {

            case c <- fmt.Sprintf("%s: %d", msg, i):

                // do nothing

            case <-quit:

                return

            }

```

 

, We can also know when it's finished . We simpley wait for it to tell us it's done.

 

That is , we recieve on the quit channel.

 

```

quit := make(chan string)

    c := boring("Joe", quit)

    for i := rand.Intn(10); i >= 0; i-- { fmt.Println(<-c) }

    quit <- "Bye!"

    fmt.Printf("Joe says: %q\n", <-quit)

 

 

```

 

 

```

select {

            case c <- fmt.Sprintf("%s: %d", msg, i):

                // do nothing

            case <-quit:

                cleanup()

                quit <- "See you!"

                return

            }

 

```

 

 

 

### Daisy Chain

 

A sequential way to pass information by dividing a task into steps. We can achieve pipelining via this method. Hence, get better throughput.

 

```

 

func f(left, right chan int) {

    left <- 1 + <-right

}

 

func main() {

    const n = 10000

    leftmost := make(chan int)

    right := leftmost

    left := leftmost

    for i := 0; i < n; i++ {

        right = make(chan int)

        go f(left, right)

        left = right

    }

    go func(c chan int) { c <- 1 }(right)

    fmt.Println(<-leftmost)

}

 

```

 

Basically this program plays a game of Chinese Whisper. The output of the program will be 10001.

 

# _Go_ Concurrency Features with System Software

 

We are going to take a look into how the Google Search will work using Concurrency.

 

## Google Search

 

- Query: A question

- Answer: A set of results.

 

We get our search results by quering Web Search, Image Search, Youtube, Maps, News etc. Then we mix the results.

 

### Fake Framework

 

We simulate the search functoin.

```

var (

    Web = fakeSearch("web")

    Image = fakeSearch("image")

    Video = fakeSearch("video")

)

 

type Search func(query string) Result

 

func fakeSearch(kind string) Search {

        return func(query string) Result {

              time.Sleep(time.Duration(rand.Intn(100)) * time.Millisecond)

              return Result(fmt.Sprintf("%s result for %q\n", kind, query))

        }

}

 

 

```

Testing the framework

 

```

func main() {

    rand.Seed(time.Now().UnixNano())

    start := time.Now()

    results := Google("golang")

    elapsed := time.Since(start)

    fmt.Println(results)

    fmt.Println(elapsed)

}

 

 

```

 

### Google Search Version 1.0

 

```

func Google(query string) (results []Result) {

    results = append(results, Web(query))

    results = append(results, Image(query))

    results = append(results, Video(query))

    return

}

 

```

 

 

### Google Search Version 2.0

 

Introducing Concurrency

 

No locks, No condition variables. No callbacks.

 

```

func Google(query string) (results []Result) {

    c := make(chan Result)

    go func() { c <- Web(query) } ()

    go func() { c <- Image(query) } ()

    go func() { c <- Video(query) } ()

 

    for i := 0; i < 3; i++ {

        result := <-c

        results = append(results, result)

    }

    return

}

 

```

 

### Google Search Versoin 2.1

 

 

Adding timeout. Don't wait for slow results.

 

```

c := make(chan Result)

    go func() { c <- Web(query) } ()

    go func() { c <- Image(query) } ()

    go func() { c <- Video(query) } ()

 

    timeout := time.After(80 * time.Millisecond)

    for i := 0; i < 3; i++ {

        select {

        case result := <-c:

            results = append(results, result)

        case <-timeout:

            fmt.Println("timed out")

            return

        }

    }

    return

 

```

 

To avoid discarding results from a slow server. We can have replicas !!

 

Hence, we shoot the results to a bunch of replicas and return whatever comes along first.

 

```

func First(query string, replicas ...Search) Result {

    c := make(chan Result)

    searchReplica := func(i int) { c <- replicas[i](query) }

    for i := range replicas {

        go searchReplica(i)

    }

    return <-c

}

```

 

```

func main() {

    rand.Seed(time.Now().UnixNano())

    start := time.Now()

    result := First("golang",

        fakeSearch("replica 1"),

        fakeSearch("replica 2"))

    elapsed := time.Since(start)

    fmt.Println(result)

    fmt.Println(elapsed)

}

 

```

 

 

### Google Search Version 3.0

 

We can reduce tail latency by using replicated search servers.

 

 

```

c := make(chan Result)

    go func() { c <- First(query, Web1, Web2) } ()

    go func() { c <- First(query, Image1, Image2) } ()

    go func() { c <- First(query, Video1, Video2) } ()

    timeout := time.After(80 * time.Millisecond)

    for i := 0; i < 3; i++ {

        select {

        case result := <-c:

            results = append(results, result)

        case <-timeout:

            fmt.Println("timed out")

            return

        }

    }

    return

 

 

 

```

 

 

### Conclusion

 

In only a few lines we have converted a _slow,sequential, failure sensitive_ program into a program that is:

1. fast

2. concurrent

3. replicated

4. robust

 

Hope this helps. In the nest post, we can look into some more advanced concurrency patterns.

 

Cheers.

 
