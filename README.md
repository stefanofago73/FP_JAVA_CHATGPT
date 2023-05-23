# FP_JAVA_CHATGPT
A set of examples of a hypothetical Functional Programming course, developed with ChatGPT... between an experiment and a joke

## WHAT INSIDE

For the examples involving functors, cofunctors, and natural transformations, the main classes and interfaces used include:

- Functor<T> and Cofunctor<T>: These are interfaces representing a type that can be mapped over in a forward or backward direction, respectively.

- FMap<F, T> and CofMap<F, T>: These are interfaces representing a function that maps a value of type F to a value of type T in a forward or backward direction, respectively.

- NaturalTransformation<F, G>: This is an interface representing a natural transformation from the functor F to the functor G.


For the examples involving monads and monad transformers, the main classes and interfaces used include:

- Monad<T>: This is an interface representing a type that supports monadic operations such as unit, bind, and map.
- Maybe<T>: This is a class representing a monad that may or may not contain a value of type T.

- IO<T>: This is a class representing a monad that performs I/O operations and returns a value of type T.

- MonadTransformer<M, T>: This is an interface representing a monad transformer that takes a monad M and transforms it into a new monad that includes additional effects.
  
  # WARNING - WHAT FOLLOW IS INCOMPLETE AND UNFILTERED CODE; IT COMES DIRECTLY FROM A CHAT SESSION
  

### FUNCTOR & COFUNCTOR

```java
  
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;


class Functor {
    public static <A, B> List<B> map(List<A> list, Function<A, B> f) {
        List<B> result = new ArrayList<>();
        for (A a : list) {
            result.add(f.apply(a));
        }
        return result;
    }
}

  class Cofunctor {
    public static <A, B> B extract(A a, Function<A, B> f) {
        return f.apply(a);
    }
}

  
public class FunctorExample {
    public static void main(String[] args) {
        List<Integer> list = Arrays.asList(1, 2, 3, 4, 5);
        List<Integer> result = Functor.map(list, x -> x * 2);
        System.out.println(result); // [2, 4, 6, 8, 10]
    }
}

  
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public class CofunctorExample {
    public static void main(String[] args) {
        List<Integer> list = Arrays.asList(1, 2, 3, 4, 5);
        Integer result = Cofunctor.extract(list, xs -> xs.stream().mapToInt(Integer::intValue).sum());
        System.out.println(result); // 15
    }
}


import java.util.function.Function;

public class Example1 {

    public static void main(String[] args) {
        // Create a functor and a cofunctor
        List<Integer> list = List.of(1, 2, 3);
        Function<String, Integer> length = String::length;

        // Map the cofunctor over the functor using the fmap method
        List<Integer> result = Functor.fmap(list, Cofunctor.cofmap(length));

        // Print the result
        System.out.println(result); // [1, 1, 1]
    }
}

import java.util.function.Function;

public class Example2 {

    public static void main(String[] args) {
        // Create a functor and two cofunctors
        List<String> list = List.of("foo", "bar", "baz");
        Function<Integer, String> toString = Object::toString;
        Function<String, Integer> length = String::length;

        // Compose the functor and cofunctor using the compose method
        Function<Integer, Integer> composed = Cofunctor.compose(length, toString);

        // Map the composed function over the functor using the fmap method
        List<Integer> result = Functor.fmap(list, composed);

        // Print the result
        System.out.println(result); // [3, 3, 3]
    }
}
```


### NATURAL TRANSFORMATION EXAMPLE

```java
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public class NaturalTransformationExample {
    public static void main(String[] args) {
        List<Integer> list = Arrays.asList(1, 2, 3, 4, 5);
        Function<Integer, Double> f = x -> x * 0.5;
        Function<Double, String> g = x -> "Value: " + x;
        
        // Construct a natural transformation between ListFunctor and StreamFunctor
        NaturalTransformation<ListFunctor, StreamFunctor> natTrans = new NaturalTransformation<ListFunctor, StreamFunctor>() {
            @Override
            public <A> StreamFunctor<A> apply(ListFunctor<A> fa) {
                return new StreamFunctor<>(fa.getList().stream().map(f));
            }
        };
        
        // Use the natural transformation to transform a ListFunctor to a StreamFunctor
        StreamFunctor<Double> stream = natTrans.apply(new ListFunctor<>(list));
        List<String> result = stream.getStream().map(g).toList();
        System.out.println(result); // [Value: 0.5, Value: 1.0, Value: 1.5, Value: 2.0, Value: 2.5]
    }
}

// Functor that represents a list of values
class ListFunctor<A> {
    private final List<A> list;

    public ListFunctor(List<A> list) {
        this.list = list;
    }

    public List<A> getList() {
        return list;
    }
}

// Functor that represents a stream of values
class StreamFunctor<A> {
    private final java.util.stream.Stream<A> stream;

    public StreamFunctor(java.util.stream.Stream<A> stream) {
        this.stream = stream;
    }

    public java.util.stream.Stream<A> getStream() {
        return stream;
    }
}

// Natural transformation between two functors
interface NaturalTransformation<F extends Functor, G extends Functor> {
    <A> G apply(F fa);
}
```


### MONAD/COMONAD EXAMPLE

```java
import java.util.Optional;

public class MonadExample {
    public static void main(String[] args) {
        Optional<Integer> optionalInt = Optional.of(5);
        Optional<Integer> result = optionalInt.flatMap(i -> {
            if (i % 2 == 0) {
                return Optional.of(i / 2);
            } else {
                return Optional.empty();
            }
        });
        System.out.println(result); // Optional.empty()
    }
}
      
import java.util.List;

public class ComonadExample {
    public static void main(String[] args) {
        List<Integer> list = List.of(1, 2, 3, 4, 5);
        List<List<Integer>> result = ListComonad.coflatMap(list, i -> ListComonad.tail(list.subList(i, i + 3)));
        System.out.println(result);
    }
}

class ListComonad {
    public static <A> List<List<A>> coflatMap(List<A> list, Function<List<A>, A> f) {
        List<List<A>> result = new ArrayList<>();
        for (int i = 0; i < list.size(); i++) {
            List<A> sublist = list.subList(i, list.size());
            List<A> head = List.of(sublist.get(0));
            List<A> tail = f.apply(sublist);
            List<A> resultSublist = new ArrayList<>();
            resultSublist.addAll(head);
            resultSublist.addAll(tail);
            result.add(resultSublist);
        }
        return result;
    }

    public static <A> List<A> tail(List<A> list) {
        if (list.isEmpty()) {
            throw new IllegalArgumentException("list cannot be empty");
        } else {
            return list.subList(1, list.size());
        }
    }
}
```

### MONAD TRANSFORMER EXAMPLES


```java
      
import java.util.Optional;
import java.util.stream.Stream;

public class Example {

    public static void main(String[] args) {
        // Create an optional that contains a stream
        Optional<Stream<Integer>> optional = Optional.of(Stream.of(1, 2, 3));

        // Transform the optional using the stream monad
        Optional<Stream<Integer>> result = MonadTransformer.transform(optional, i -> Stream.of(i * 2));

        // Print the result
        System.out.println(result); // Optional[[2, 4, 6]]
    }
}

interface MonadTransformer<M1, M2> {
    static <A, B> Function<M1, M2> lift(Function<A, B> f, Monad<M1> m1, Monad<M2> m2) {
        return a -> m1.flatMap(m2.lift(f).apply(a), m2);
    }

    static <A, B> M2 transform(M1 m1, Function<A, M2> f) {
        return lift(f, new Monad<M1>() {}, new Monad<M2>() {}).apply(m1);
    }
}

interface Monad<M> {
    <A, B> M flatMap(Function<A, M> f, M m);
    <A, B> Function<M, M> lift(Function<A, B> f);
}

class OptionalMonad implements Monad<Optional<?>> {
    @Override
    public <A, B> Optional<?> flatMap(Function<A, Optional<?>> f, Optional<?> m) {
        return m.flatMap(f);
    }

    @Override
    public <A, B> Function<Optional<?>, Optional<?>> lift(Function<A, B> f) {
        return m -> m.map(f);
    }
}

class StreamMonad implements Monad<Stream<?>> {
    @Override
    public <A, B> Stream<?> flatMap(Function<A, Stream<?>> f, Stream<?> m) {
        return m.flatMap(a -> f.apply((A) a));
    }

    @Override
    public <A, B> Function<Stream<?>, Stream<?>> lift(Function<A, B> f) {
        return m -> m.map(f);
    }
}


import java.util.Optional;
import java.util.stream.Stream;

public class Example {

    public static void main(String[] args) {
        // Create an optional that contains a stream
        Optional<Stream<Integer>> optional = Optional.of(Stream.of(1, 2, 3));

        // Transform the optional using the stream monad
        Optional<Stream<Integer>> result = MonadTransformer.transform(optional, i -> Stream.of(i * 2));

        // Print the result
        System.out.println(result); // Optional[[2, 4, 6]]
    }
}

interface MonadTransformer<M1, M2> {
    static <A, B> Function<M1, M2> lift(Function<A, B> f, Monad<M1> m1, Monad<M2> m2) {
        return a -> m1.flatMap(m2.lift(f).apply(a), m2);
    }

    static <A, B> M2 transform(M1 m1, Function<A, M2> f) {
        return lift(f, new Monad<M1>() {}, new Monad<M2>() {}).apply(m1);
    }
}

interface Monad<M> {
    <A, B> M flatMap(Function<A, M> f, M m);
    <A, B> Function<M, M> lift(Function<A, B> f);
}

class OptionalMonad implements Monad<Optional<?>> {
    @Override
    public <A, B> Optional<?> flatMap(Function<A, Optional<?>> f, Optional<?> m) {
        return m.flatMap(f);
    }

    @Override
    public <A, B> Function<Optional<?>, Optional<?>> lift(Function<A, B> f) {
        return m -> m.map(f);
    }
}

class StreamMonad implements Monad<Stream<?>> {
    @Override
    public <A, B> Stream<?> flatMap(Function<A, Stream<?>> f, Stream<?> m) {
        return m.flatMap(a -> f.apply((A) a));
    }

    @Override
    public <A, B> Function<Stream<?>, Stream<?>> lift(Function<A, B> f) {
        return m -> m.map(f);
    }
}
```


### LIST COMPREHENSION


```java


import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ListComprehensionExample {
    public static void main(String[] args) {
        List<Integer> result = IntStream.rangeClosed(1, 10)
                .filter(x -> x % 2 == 0)
                .map(x -> x * x)
                .boxed()
                .collect(Collectors.toList());
        
        System.out.println(result); // [4, 16, 36, 64, 100]
    }
}


import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class ListComprehensionExample {
    public static void main(String[] args) {
        List<Integer> list1 = List.of(1, 2);
        List<Integer> list2 = List.of(3, 4);
        List<String> result = new ArrayList<>();
        
        for (Integer i : list1) {
            for (Integer j : list2) {
                result.add("(" + i + "," + j + ")");
            }
        }
        
        System.out.println(result); // [(1,3), (1,4), (2,3), (2,4)]
        
        // Alternatively, using list comprehension:
        List<String> result2 = list1.stream()
                .flatMap(i -> list2.stream().map(j -> "(" + i + "," + j + ")"))
                .collect(Collectors.toList());
        
        System.out.println(result2); // [(1,3), (1,4), (2,3), (2,4)]
    }
}



import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ListComprehensionExample {
    public static void main(String[] args) {
        int n = 20;
        List<Integer> primes = IntStream.rangeClosed(2, n)
                .filter(x -> IntStream.rangeClosed(2, (int) Math.sqrt(x))
                        .allMatch(y -> x % y != 0))
                .boxed()
                .collect(Collectors.toList());
        
```

FOR_COMPREHENSION JAVA EXAMPLE


```java

import java.util.Arrays;
import java.util.List;

public class ForComprehensionExample {
    public static void main(String[] args) {
        List<Integer> list1 = Arrays.asList(1, 2, 3);
        List<Integer> list2 = Arrays.asList(4, 5, 6);
        
        List<Integer> result = forComp(list1, x ->
                forComp(list2, y ->
                        List.of(x + y)));
        
        System.out.println(result); // [5, 6, 7, 6, 7, 8, 7, 8, 9]
    }
    
    public static <A, B> List<B> forComp(List<A> list, Function<A, List<B>> f) {
        List<B> result = new ArrayList<>();
        for (A a : list) {
            result.addAll(f.apply(a));
        }
        return result;
    }
}


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ForComprehensionExample {
    public static void main(String[] args) {
        List<Integer> list1 = Arrays.asList(1, 2, 3);
        List<String> list2 = Arrays.asList("a", "b", "c");
        
        List<Pair<Integer, String>> result = forComp(list1, x ->
                forComp(list2, y ->
                        List.of(new Pair<>(x, y))));
        
        System.out.println(result); // [(1,a), (1,b), (1,c), (2,a), (2,b), (2,c), (3,a), (3,b), (3,c)]
    }
    
    public static <A, B> List<Pair<A, B>> forComp(List<A> list1, Function<A, List<B>> f) {
        List<Pair<A, B>> result = new ArrayList<>();
        for (A a : list1) {
            for (B b : f.apply(a)) {
                result.add(new Pair<>(a, b));
            }
        }
        return result;
    }
    
    public static class Pair<A, B> {
        private final A a;
        private final B b;
        
        public Pair(A a, B b) {
            this.a = a;
            this.b = b;
        }
        
        public A getA() {
            return a;
        }
        
        public B getB() {
            return b;
        }
        
        @Override
        public String toString() {
            return "(" + a + "," + b + ")";
        }
    }
}


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ForComprehensionExample {
    public static void main(String[] args) {
        List<Integer> list = Arrays.asList(1, 2, 3, 4, 5);
        
        List<Integer> result = forComp(list, x ->
                x % 2 == 0 ? List.of(x) : List.of());
        
        System.out.println(result); // [2, 4]
    }
    
    public static <A, B> List<B> forComp(List<A> list, Function<A, List<B>> f) {
        List<B> result = new ArrayList<>();
        for (A a : list) {
            result.addAll(f.apply(a));
        }
        return result;
    }
}

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ForComprehensionExample {
    public static void main(String[] args) {
        List<Integer> list1 = Arrays.asList(1, 2, 3);
        List<Integer> list2 = Arrays.asList(4, 5, 6);
        
        List<String> result = forComp(list1, x ->
                forComp(list2, y ->
                        List.of("(" + x + "," + y + ")")));
        
        System.out.println(result); // [(1,4), (1,5), (1,6), (2,4), (2,5), (2,6), (3,4), (3,5), (3,6)]
    }
    
    public static <A, B> List<B> forComp(List<A> list, Function<A, List<B>> f) {
        List<B> result = new ArrayList<>();
        for (A a : list) {
            result.addAll(f.apply(a));
        }
        return result;
    }
}


import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ForComprehensionExample {
    public static void main(String[] args) {
        List<Pair<String, Integer>> list = Arrays.asList(
                new Pair<>("a", 1),
                new Pair<>("b", 2),
                new Pair<>("a", 3),
                new Pair<>("b", 4));
        
        Map<String, List<Integer>> result = forComp(list, p ->
                forComp(List.of(p.getA()), k ->
                        Map.of(k, List.of(p.getB()))))
                .stream()
                .reduce(new HashMap<>(), (acc, m) -> {
                    for (String k : m.keySet()) {
                        if (acc.containsKey(k)) {
                            acc.get(k).addAll(m.get(k));
                        } else {
                            acc.put(k, m.get(k));
                        }
                    }
                    return acc;
                });
        
        System.out.println(result); // {a=[1, 3], b=[2, 4]}
    }
    
    public static <A, B> List<B> forComp(List<A> list, Function<A, List<B>> f) {
        List<B> result = new ArrayList<>();
        for (A a : list) {
            result.addAll(f.apply(a));
        }
        return result;
    }
}


import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class ForComprehensionExample {
    public static void main(String[] args) {
        List<Integer> list1 = Arrays.asList(1, 2, 3);
        List<Integer> list2 = Arrays.asList(4, 5, 6);
        List<Integer> list3 = Arrays.asList(7, 8, 9);
        
        List<Integer> result = forComp(list1.stream(), x ->
                forComp(list2.stream(), y ->
                        forComp(list3.stream(), z ->
                                Stream.of(x, y, z)
                                        .collect(Collectors.toList()))))
                .collect(Collectors.toList());
        
        System.out.println(result); // [1, 4, 7, 1, 4, 8, 1, 4, 9, 1, 5, 7, 1, 5, 8, 1, 5, 9, 1, 6, 7, 1, 6, 8, 1, 6, 9, 2, 4, 7, 2, 4, 8, 2, 4, 9, 2, 5, 7, 2, 5, 8, 2, 5, 9, 2, 6, 7, 2, 6, 8, 2, 6, 9, 3, 4, 7, 3, 4, 8, 3, 4, 9, 3, 5, 7, 3, 5, 8, 3, 5, 9, 3, 6, 7, 3, 6, 8, 3, 6, 9]
    }
    
    public static <A, B> Stream<B> forComp(Stream<A> stream, Function<A, Stream<B>> f) {
        return stream.flatMap(a -> f.apply(a));
    }
}
```


### INJECTION EXAMPLE


```java


public class EitherMonadExample {

    public static void main(String[] args) {
        Either<String, Integer> either1 = Either.right(5);
        Either<String, Integer> either2 = Either.left("Error occurred");

        String result1 = either1.map(i -> i.toString()).getOrElse("default");
        System.out.println(result1); // "5"

        String result2 = either2.map(i -> i.toString()).getOrElse("default");
        System.out.println(result2); // "default"
    }
}


import java.util.function.Function;

public class ReaderMonadExample {

    public static void main(String[] args) {
        Reader<String, Integer> reader = new Reader<>(String::length);

        Function<String, Integer> func = reader.run();
        int length = func.apply("Hello, world!");
        System.out.println(length); // 13
    }
}

class Reader<E, A> {
    private final Function<E, A> func;

    public Reader(Function<E, A> func) {
        this.func = func;
    }

    public Function<E, A> run() {
        return func;
    }

    public <B> Reader<E, B> map(Function<A, B> g) {
        return new Reader<>(func.andThen(g));
    }

    public <B> Reader<E, B> flatMap(Function<A, Reader<E, B>> g) {
        return new Reader<>(e -> g.apply(func.apply(e)).run().apply(e));
    }
}


public class StateMonadExample {

    public static void main(String[] args) {
        State<Integer, String> state1 = State.of(0);
        State<Integer, String> state2 = state1.flatMap(x -> State.of(x + 1));

        Pair<Integer, String> result1 = state1.run(5);
        Pair<Integer, String> result2 = state2.run(5);

        System.out.println(result1.getLeft() + " : " + result1.getRight()); // 0 : null
        System.out.println(result2.getLeft() + " : " + result2.getRight()); // 1 : null
    }
}

class State<S, A> {

    private final Function<S, Pair<A, S>> func;

    private State(Function<S, Pair<A, S>> func) {
        this.func = func;
    }

    public static <S, A> State<S, A> of(A value) {
        return new State<>(s -> new Pair<>(value, s));
    }

    public Pair<A, S> run(S state) {
        return func.apply(state);
    }

    public <B> State<S, B> map(Function<A, B> f) {
        return flatMap(a -> State.of(f.apply(a)));
    }

    public <B> State<S, B> flatMap(Function<A, State<S, B>> f) {
        return new State<>(s -> {
            Pair<A, S> p1 = func.apply(s);
            return f.apply(p1.getLeft()).func.apply(p1.getRight());
        });
    }
}

class Pair<L, R> {
    private final L left;
    private final R right;

    public Pair(L


public class MaybeMonad<T> implements Monad<T, MaybeMonad<?>> {
    private final T value;
    
    private MaybeMonad(T value) {
        this.value = value;
    }
    
    public static <T> MaybeMonad<T> just(T value) {
        return new MaybeMonad<>(value);
    }
    
    public static <T> MaybeMonad<T> nothing() {
        return new MaybeMonad<>(null);
    }
    
    public <R> MaybeMonad<R> flatMap(Function<T, MaybeMonad<R>> f) {
        return value != null ? f.apply(value) : MaybeMonad.nothing();
    }
    
    public <R> MaybeMonad<R> map(Function<T, R> f) {
        return value != null ? MaybeMonad.just(f.apply(value)) : MaybeMonad.nothing();
    }
}



public class StateMonad<S, T> implements Monad<T, StateMonad<S, ?>> {
    private final Function<S, Tuple<T, S>> function;
    
    private StateMonad(Function<S, Tuple<T, S>> function) {
        this.function = function;
    }
    
    public static <S, T> StateMonad<S, T> of(Function<S, Tuple<T, S>> function) {
        return new StateMonad<>(function);
    }
    
    public <R> StateMonad<S, R> flatMap(Function<T, StateMonad<S, R>> f) {
        return StateMonad.of(s -> {
            Tuple<T, S> tuple = function.apply(s);
            return f.apply(tuple.getFirst()).function.apply(tuple.getSecond());
        });
    }
    
    public <R> StateMonad<S, R> map(Function<T, R> f) {
        return StateMonad.of(s -> {
            Tuple<T, S> tuple = function.apply(s);
            return new Tuple<>(f.apply(tuple.getFirst()), tuple.getSecond());
        });
    }
    
    public Tuple<T, S> run(S state) {
        return function.apply(state);
    }
}

```


















