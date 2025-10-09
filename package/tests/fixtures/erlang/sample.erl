%% module declaration
-module(my_module).
-export([say_hello/0, main/0, process_data/1]).

%% imports - multiple modules with various functions
-import(lists, [map/2, filter/2, foldl/3]).
-import(io, [format/1, format/2]).
-import(string, [concat/2, uppercase/1]).
-import(maps, [new/0, put/3]).
-import(erlang, [now/0]).

%% function
say_hello() ->
    io:format("Hello from Erlang!~n").

process_data(Data) ->
    lists:map(fun(X) -> X * 2 end, Data).

main() ->
    say_hello().  %% call

%% execute main
main().