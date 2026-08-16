from __future__ import annotations

from dataclasses import asdict, dataclass

MEBIBYTE = 1024 ** 2
@dataclass(frozen=True)
class ModelConfig:
    """
    Central configuration for MiniLexGPT architecture.
  
    The class is frozen so that its values cannot be changed 
    accidentally after creation.
    """

    # Tokenizer and sequence settings
    vocab_size: int = 4096
    context_length: int = 128

    # Transformer dimensions
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    ffn_hidden_size: int = 384

    # Regularization and normalization
    dropout: float = 0.1
    rms_norm_epsilon: float = 1e-5

    #Rotary Postion Embedding
    rope_base: float = 10_000.0
    
    initializer_std: float = 0.02
    # Modern decoder-only design choices
    use_bias: bool = False
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        """
        
        Validate the configuration immediately after creation.
        """

        positive_integer_fields = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_hidden_size": self.ffn_hidden_size,
        }

        for field_name, field_value in positive_integer_fields.items():
            if field_value <= 0:
                raise ValueError(
                    f"{field_name} must be greater than zero,"
                    f"but received {field_value}."
                 )

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
                f"Received d_model={self.d_model} and"
                f"num_heads={self.num_heads}."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0.0 <= dropout < 1.0."
            )

        if self.rms_norm_epsilon <= 0:
            raise ValueError(
                "rms_norm_epsilon must be grater than zero."
            )
    
        if self.rope_base <= 0:
            raise ValueError(
                "rope_base must be greater than zero."
            )
      
        if self.num_kv_heads <= 0:
            raise ValueError(
                "num_kv_heads must be greater than zero."
            )

        if self.num_kv_heads > self.num_heads:
            raise ValueError(
                "num_kv_heads cannot be greater than "
                "num_heads."
            )
     
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_heads must be divisible by "
                "num_kv_heads."
            )
       
        if self.initializer_std <= 0:
            raise ValueError(
                "initializer_std must be greater than zero."
            )

    @property
    def head_dimension(self) -> int:
        """

        Return the vector dimension used by one attention head.
        """

        return self.d_model // self.num_heads

    @property
    def query_dimension(self) -> int:
        """
        Total dimension of all Query heads.
        """

        return self.num_heads * self.head_dimension

    @property
    def kv_dimension(self) -> int:
        """
        Total dimension of all key or value heads.
        """
 
        return self.num_kv_heads * self.head_dimension

    @property
    def query_heads_per_kv_head(self) -> int:
        """
        
        Number of Query heads sharing one KV head.
        """
        return self.num_heads // self.num_kv_heads
    def parameter_breakdown(self) -> dict[str, int]:
        """
        Return the vector dimension used by one attention head.
        """
    
        return self.d_model // self.num_heads

    def parameter_breakdown(self) -> dict[str, int]:
        """
        Estimate the number of trainable parameters.
        
        The calculation assumes:
        - bias-free linear projections
        - RMSNorm with one trainable vector
        - four attention matrices: Q, K, V and output
        - three SwiGLU matrices: gate, up and down
        - RoPE with no trainable parameters
        """

        token_embedding = self.vocab_size * self.d_model

        bias_enabled = int(self.use_bias)
        
        query_projection = (
            self.d_model * self.query_dimension
            + bias_enabled * self.query_dimension
        )
        
        key_projection = (
            self.d_model * self.kv_dimension
            + bias_enabled * self.kv_dimension
        )

        value_projection = (
            self.d_model * self.kv_dimension
            + bias_enabled * self.kv_dimension
        )
        
        attention_output_projection = (
            self.query_dimension * self.d_model
            + bias_enabled * self.d_model
        )
        attention_per_block = (
            query_projection
            + key_projection
            + value_projection
            + attention_output_projection
        )

        swiglu_per_block = (
            3 * self.d_model * self.ffn_hidden_size
        )
        
        rmsnorm_per_block = 2 * self.d_model
   
        transformer_block = (
            attention_per_block
            + swiglu_per_block
            + rmsnorm_per_block
            
        )
        all_transformer_blocks = (
            self.num_layers * transformer_block
        )

        final_rmsnorm = self.d_model

        if self.tie_embeddings:
            language_model_head = 0
        else:
            language_model_head = (
                self.vocab_size * self.d_model
            )
        total = (
            token_embedding
            + all_transformer_blocks
            + final_rmsnorm
            + language_model_head
        )
        
        return {
            "token_embedding": token_embedding,
            "attention_per_block": attention_per_block,
            "swiglu_per_block": swiglu_per_block,
            "rmsnorm_per_block": rmsnorm_per_block,
            "one_transformer_block": transformer_block,
            "all_transformer_blocks": all_transformer_blocks,
            "final_rmsnorm": final_rmsnorm,
            "language_model_head": language_model_head,
            "total": total,
        }

    def parameter_memory_mib(self) -> float:
        """
        
        Estimate float32 memory used by model parameters omly.
        """

        total_parameters = self.parameter_breakdown()["total"]
        number_of_bytes = total_parameters * 4
        
        return number_of_bytes / MEBIBYTE

    def adamw_training_memory_mib(self) -> float:
        """
        Estimate parameter-related float32 training memory.

        Approximation per parameter:
        - model weight: 4 bytes
        - gradient: 4 bytes
        - Adam first moment: 4 bytes
        - Adam second moment: 4 bytes

        Activations, input batchs and framwork overhead are 
        not included in this estimate.
        """

        total_parameters = self.parameter_breakdown()["total"]
        number_of_bytes = total_parameters * 16

        return number_of_bytes / MEBIBYTE

    def to_dictionary(self) -> dict[str, object]:
        """

        Return the configuration as a normal python dictionary.
        """
        return asdict(self)
