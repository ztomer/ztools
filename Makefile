.PHONY: build test fmt install clean ci

# The GATE lives in ONE place: .gatesrc (GOH_CI_STEPS), which `make ci` and the
# pre-push hook both read. Do not re-state any step here — three targets used to
# duplicate it, and the coverage target re-declared the floor (94 vs 95) as a
# second writable copy of a number the gate enforces. A target that shadows a
# gate is a way for the gate to be wrong in two places at once.
#
#   make ci      the whole gate of record (what pre-push runs)
#   make test    just the Rust suite, for a fast inner loop
#   make fmt     rewrite formatting (the gate CHECKS it; run this first)
#   make build   debug binary
#   make install ./install.sh — release build, platform gate, symlinks

build: ## Build the ztools binary
	cargo build --manifest-path rust/Cargo.toml

test: ## Run the Rust test suite
	cargo test --manifest-path rust/Cargo.toml --all-features

fmt: ## Format all Rust code
	cargo fmt --manifest-path rust/Cargo.toml --all

install: ## Build and install the binaries to $(brew --prefix)/bin
	./install.sh

ci: ## The gate of record — same list the pre-push hook runs
	@# Step list lives in .gatesrc (GOH_CI_STEPS); this only delegates.
	"$${GOH_DIR:-$$HOME/Projects/gates_of_heck}/gates/local_ci.sh" .

clean:
	cargo clean --manifest-path rust/Cargo.toml
