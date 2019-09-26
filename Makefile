swaption_pricing.so: libswaption_pricing.so
	ln -fs $< $@

.PHONY:
clean:
	rm *.so

.PHONY:
run:
	python3 python_pricing.py --path 10000 --grid 200 --seed 1234

libswaption_pricing.so:
	clang++ -std=c++17 main.cpp -I/usr/local/include `python3-config --cflags --ldflags` -I/usr/local/include/eigen3 -march=native  -shared -o libswaption_pricing.so
