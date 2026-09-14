.PHONY: build test fmt lint gate coverage ci install clean

build: ## Build the ztools binary
	cargo build --manifest-path rust/Cargo.toml

test: ## Run the Rust test suite
	cargo test --manifest-path rust/Cargo.toml --all-features

fmt: ## Format all Rust code
	cargo fmt --manifest-path rust/Cargo.toml --all

lint: ## The shared house Rust gate (fmt + clippy -D warnings + no #[allow])
	"$${GOH_DIR:-$$HOME/Projects/gates_of_heck}/gates/rust_gate.sh" . rust

gate: ## The structural house gates (emoji, file length, markers, shell lint, secrets)
	"$${GOH_DIR:-$$HOME/Projects/gates_of_heck}/gates/structural.sh" --full

coverage: ## Enforce the coverage floor
	"$${GOH_DIR:-$$HOME/Projects/gates_of_heck}/gates/coverage_gate.sh" --lang rust --floor 95 rust

install: ## Build and install the binaries to $(brew --prefix)/bin
	./install.sh

ci: ## The gate of record -- same list the pre-push hook runs
	@# Step list lives in .gatesrc (GOH_CI_STEPS); this only delegates.
	"$${GOH_DIR:-$$HOME/Projects/gates_of_heck}/gates/local_ci.sh" .

clean:
	cargo clean --manifest-path rust/Cargo.toml
