package com.carbonpilot;

import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;

import java.util.List;

public class CloudSimRunner {
    public static void main(String[] args) {
        CloudSimPlus simulation = new CloudSimPlus();

        List<Pe> peList = List.of(
            new PeSimple(1000),
            new PeSimple(1000),
            new PeSimple(1000),
            new PeSimple(1000)
        );

        Host host = new HostSimple(16384, 10000, 1000000, peList);

        Datacenter datacenter = new DatacenterSimple(simulation, List.of(host));

        DatacenterBroker broker = new DatacenterBrokerSimple(simulation);

        Vm vm = new VmSimple(1000, 2);
        vm.setRam(4096).setBw(1000).setSize(10000);

        Cloudlet cloudlet = new CloudletSimple(10000, 2);
        cloudlet.setFileSize(300).setOutputSize(300);

        broker.submitVm(vm);
        broker.submitCloudlet(cloudlet);

        simulation.start();

        System.out.println("CloudSim Plus Simulation Completed");
        System.out.println("Cloudlets: " + broker.getCloudletFinishedList().size());
        System.out.println("VMs: " + broker.getVmCreatedList().size());
    }
}