from neuron import Neuron
from synapse import Synapse
    
    # Small neural network
# Neuron A ─────[ Synapse AC ]─────┐
#                 Weight: 0.50     │
#                 Excitatory (+)   │
#                                  ├───> Neuron C
# Neuron B ─────[ Synapse BC ]─────┘
#                 Weight: 0.25
#                 Inhibitory (−)

def main():
    neuronA = Neuron(state=0.4, threshold=0.6)
    neuronB = Neuron(state=0.3, threshold=0.4)
    neuronC = Neuron(state=0.2, threshold=0.35) #output neuron
    
    # IMPO: Its synapses that classify as excitatory or inhibitory, not the neurons themselves.
    
    synapseAC = Synapse(weight=0.5, conn_type=0) #excitatory synapse from A to C
    synapseBC = Synapse(weight=0.25, conn_type=1) #inhibitory synapse from B to C
    
    
    #Neuron A recieves excitation=0.5, inhibition=0.2
    outputA = neuronA.update(excitation=0.5, inhibition=0.2)
    
    #Neuron B recieves excitation=0.4, inhibition=0.1
    outputB = neuronB.update(excitation=0.4, inhibition=0.1)
    
    ACtransmission = synapseAC.transmit(outputA)
    BCtransmission = synapseBC.transmit(outputB)
    
    
    outputC = neuronC.update(excitation=ACtransmission[0], inhibition=BCtransmission[1])
    
    
    print(f"Neuron A output: {outputA}, New state: {neuronA.state}")
    print(f"Neuron B output: {outputB}, New state: {neuronB.state}")
    
    print(f"Total excitation to C: {ACtransmission[0]}, total inhibition to C: {BCtransmission[1]}")
    
    print(f"Neuron C output: {outputC}, New state: {neuronC.state}")
    
    
    
if __name__ == "__main__":
    main()