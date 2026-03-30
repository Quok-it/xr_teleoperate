import sys
sys.path.insert(0, '/home/quokka')

# First, let's find any mcap or protobuf packages
import subprocess
result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
print("=== Installed packages with mcap/proto ===")
for line in result.stdout.split('\n'):
    if 'mcap' in line.lower() or 'proto' in line.lower():
        print(line)

print("\n\n")

# Try to read the MCAP file and extract schema
from mcap.reader import make_reader

with open('/home/quokka/data-newtheory-format/episode_0389/data.mcap', 'rb') as f:
    reader = make_reader(f)
    summary = reader.get_summary()

    print("=== Schemas ===")
    for schema_id, schema in summary.schemas.items():
        print(f"Schema ID: {schema_id}")
        print(f"  Name: {schema.name}")
        print(f"  Encoding: {schema.encoding}")
        print(f"  Data length: {len(schema.data)} bytes")
        print(f"  Data (raw): {schema.data[:500]}")
        print(f"  Data (decoded): {schema.data.decode('utf-8', errors='replace')[:2000]}")
        print()

    print("=== Channels ===")
    for channel_id, channel in summary.channels.items():
        print(f"Channel ID: {channel_id}, Topic: {channel.topic}, Schema ID: {channel.schema_id}, Encoding: {channel.message_encoding}")

print("\n\n=== Trying to decode messages ===")

# Try with protobuf decoder
try:
    from mcap_protobuf.decoder import DecoderFactory
    from mcap.reader import make_reader as mr2

    with open('/home/quokka/data-newtheory-format/episode_0389/data.mcap', 'rb') as f:
        reader = mr2(f, decoder_factories=[DecoderFactory()])
        for i, (schema, channel, message, decoded) in enumerate(reader.iter_decoded_messages()):
            print(f"Message {i}: topic={channel.topic}")
            print(f"  Decoded type: {type(decoded)}")
            print(f"  Decoded: {decoded}")
            if hasattr(decoded, 'DESCRIPTOR'):
                for field in decoded.DESCRIPTOR.fields:
                    print(f"  Field: {field.name} = {getattr(decoded, field.name)}")
            if i >= 15:
                break
except Exception as e:
    print(f"Protobuf decoder error: {e}")
    import traceback
    traceback.print_exc()

# Also try raw message inspection
print("\n\n=== Raw message inspection ===")
with open('/home/quokka/data-newtheory-format/episode_0389/data.mcap', 'rb') as f:
    reader = make_reader(f)
    for i, (schema, channel, message) in enumerate(reader.iter_messages()):
        if i < 2 or (channel.topic.endswith('/leader') and 'ee' in channel.topic and i < 100):
            print(f"Message {i}: topic={channel.topic}, size={len(message.data)} bytes")
            # Try to decode as protobuf manually
            import struct
            data = message.data
            print(f"  Raw hex: {data[:100].hex()}")
            # Parse protobuf wire format
            pos = 0
            while pos < len(data) and pos < 200:
                if pos >= len(data):
                    break
                byte = data[pos]
                field_num = byte >> 3
                wire_type = byte & 0x07
                pos += 1
                if wire_type == 2:  # Length-delimited
                    length = data[pos]
                    pos += 1
                    field_data = data[pos:pos+length]
                    if length % 4 == 0 and length >= 4:
                        floats = struct.unpack(f'<{length//4}f', field_data)
                        print(f"  Field {field_num} (bytes, len={length}): floats={floats}")
                    else:
                        print(f"  Field {field_num} (bytes, len={length}): {field_data[:50].hex()}")
                    pos += length
                elif wire_type == 0:  # Varint
                    val = 0
                    shift = 0
                    while pos < len(data):
                        b = data[pos]
                        val |= (b & 0x7f) << shift
                        pos += 1
                        shift += 7
                        if not (b & 0x80):
                            break
                    print(f"  Field {field_num} (varint): {val}")
                elif wire_type == 5:  # 32-bit
                    val = struct.unpack('<f', data[pos:pos+4])[0]
                    print(f"  Field {field_num} (float32): {val}")
                    pos += 4
                elif wire_type == 1:  # 64-bit
                    val = struct.unpack('<d', data[pos:pos+8])[0]
                    print(f"  Field {field_num} (float64): {val}")
                    pos += 8
                else:
                    print(f"  Field {field_num} (wire_type={wire_type}): unknown")
                    break
            print()
            if 'ee' in channel.topic and 'leader' in channel.topic:
                break
